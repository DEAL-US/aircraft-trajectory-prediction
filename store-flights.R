library(openSkies)
library(progress)
library(torch)

load("./frankfurt-861-flights.RData") 
euclidean <- function(a, b) sqrt(sum((a - b)^2))
phases_map <- list("Ground"=0, "Climb"=1, "Cruise"=2, "Descent"=3, "Level"=4)

number_of_points <- c(5, 6, 7, 8, 9, 10)
max_future_predictions <- 20
timeResolution <- 30
for (timeResMultiplier in 1:4){
    finalTimeResolution <- timeResolution * timeResMultiplier
    features <- c("latitude", "longitude", "true_track", "baro_altitude", "geo_altitude", "vertical_rate")
    
    # For every possible number of former points being used
    for(nop in number_of_points){
        number_samples <- 0
        xs_list <- list()
        ys_list <- list()
        extra_list <- list()
        # For each flight
        pb <- progress_bar$new(total = length(flights))
        nf <- 0
        for(flight in flights){
            times <- flight$state_vectors$get_values("requested_time")
            altitudes <- flight$state_vectors$get_values("geo_altitude")
            verticalRates <- flight$state_vectors$get_values("vertical_rate")
            speeds <- flight$state_vectors$get_values("velocity")
            phases <- data.frame(times, altitudes, verticalRates, speeds)
            phases <- na.omit(phases)
            phases$index <- as.numeric(rownames(phases))
            phases$phase <- findFlightPhases(phases$times, phases$altitudes, phases$verticalRates, phases$speeds)
            
            num_sv <- length(flight$state_vectors$state_vectors)
            i_last_sv <- num_sv
            last_sv = flight$state_vectors$state_vectors[[num_sv]]
            end_lat <- last_sv$latitude
            end_lon <- last_sv$longitude
            if(is.null(end_lat) | is.null(end_lon)){
                next
            }
            
            pb$tick()
            # For every way of taking state vectors in intervals
            for(offset in 1:timeResMultiplier){
                positions <- seq(1, length(flight$state_vectors$state_vectors),  )+(offset-1)
                # Last position can be outside bounds because of the offset
                if(positions[length(positions)] > num_sv){
                    positions <- positions[1:length(positions)-1]
                }
                state_vectors <- flight$state_vectors$state_vectors[positions]
                if(length(state_vectors) < (nop+1)){
                    break
                }
                # For every possible starting state vector in the flight
                for(i in 1:(length(state_vectors)-nop)){
                    all_non_null <- TRUE
                    # We check that all the points are not null and there is at least one next point to predict
                    for(j in 0:(nop)){
                        if(is.null(state_vectors[[i+j]])){
                            all_non_null <- FALSE
                            break
                        }
                    }
                    if(all_non_null){
                        number_samples <- number_samples + 1
                        # Taking the data for prediction
                        x <- torch_zeros(nop, length(features))
                        for(j in 0:(nop-1)){
                            positions <- i+j
                            sv <- state_vectors[[positions]]
                            values <- sapply(features, function(feature){sv[[feature]]})
                            values[sapply(values, is.null)] <- -999999
                            x[j+1] <- unlist(values)
                        }
                        phase <- phases[phases$index==positions,"phase"]
                        if(length(phase) < 1){
                            phase <- -1
                        } else {
                            phase <- get(phase, phases_map)
                        }
                        last_x_lat <- sv$latitude
                        last_x_lon <- sv$longitude
                        distance_to_end <- euclidean(c(end_lat, end_lon), c(last_x_lat, last_x_lon))
                        extra <- torch_tensor(c(end_lat, end_lon, distance_to_end, phase))
                        if(length(extra)<4){
                            print("PROBLEM")
                            print(extra)
                            print(end_lat)
                            print(end_lon)
                            print(distance_to_end)
                            print(phase)
                        }
                        
                        y = torch_zeros(max_future_predictions, length(features))
                        for(j in 1:max_future_predictions) {
                            position <- i+nop+j-1
                            if(position <= length(state_vectors)){
                                sv <- state_vectors[[position]]
                                values <- sapply(features, function(feature){sv[[feature]]})
                                values[sapply(values, is.null)] <- -999999
                                y[j] <- unlist(values)
                            } else {
                                y[j] <- unlist(sapply(features, function(feature){-1}))
                            }
                            xs_list[[number_samples]] <- x
                            ys_list[[number_samples]] <- y
                            extra_list[[number_samples]] <- extra
                        }
                    }
                }
            }
        }
        print(number_samples)
        xs <- torch_zeros(number_samples, nop, length(features))
        ys <- torch_zeros(number_samples, max_future_predictions, length(features))
        extras <- torch_zeros(number_samples, 4)
        for(i in 1:number_samples){
            xs[i] <- xs_list[[i]]
            ys[i] <- ys_list[[i]]
            extras[i] <- extra_list[[i]]
        }
        folder_xs <- paste0(".training-data/training-data-", finalTimeResolution)
        folder_ys <- paste0(".training-data/training-data-", finalTimeResolution)
        folder_extras <- paste0(".training-data/training-data-", finalTimeResolution)
        if(!dir.exists(folder_xs)){
            dir.create(folder_xs)
        }
        if(!dir.exists(folder_ys)){
            dir.create(folder_ys)
        }
        if(!dir.exists(folder_extras)){
            dir.create(folder_extras)
        }
        folder_xs <- paste0(folder_xs, "/", nop)
        folder_ys <- paste0(folder_ys, "/", nop)
        folder_extras <- paste0(folder_extras, "/", nop)
        if(!dir.exists(folder_xs)){
            dir.create(folder_xs)
        }
        if(!dir.exists(folder_ys)){
            dir.create(folder_ys)
        }
        if(!dir.exists(folder_extras)){
            dir.create(folder_extras)
        }
        file_xs <- paste0(folder_xs, "/xs.pt")
        file_ys <- paste0(folder_ys, "/ys.pt")
        file_extras <- paste0(folder_extras, "/extra.pt")
        torch_save(xs, file_xs)
        torch_save(ys, file_ys)
        torch_save(extras, file_extras)
    }
}

number_sv = 0
number_vertical_rate = 0

for(flight in flights){
    svs <- flight$state_vectors$state_vectors
    number_sv = number_sv + length(svs)
    for(sv in svs){
        if(!is.null(sv$vertical_rate)){
            number_vertical_rate = number_vertical_rate + 1
        }
    }
}
print(number_sv)
print(number_vertical_rate)

times <- flights[[100]]$state_vectors$get_values("requested_time")
altitudes <- flights[[100]]$state_vectors$get_values("geo_altitude")
verticalRates <- flights[[100]]$state_vectors$get_values("vertical_rate")
speeds <- flights[[100]]$state_vectors$get_values("velocity")

data <- na.omit(data)
data$index <- as.numeric(rownames(data))
data$phase <- findFlightPhases(data$times, data$altitudes, data$verticalRates, data$speeds)
