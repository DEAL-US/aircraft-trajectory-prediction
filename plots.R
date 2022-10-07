library(gridExtra)
library(grid)
library(dplyr)
library(viridis)
library(ggplot2)
library(extrafont)

data = read.csv2("./test-results/results_summary.csv", dec=".", colClasses = c("character", rep("numeric", 8)))
data$Prediction.position = data$Prediction.position+1
data = data[data$Prediction.position<=5,]

data_diff_abs = dplyr::filter(data, !grepl('epochs|th|additional|extra|turns|spline|weighted|attention', Model))
data_net_base = dplyr::filter(data, !grepl('th|additional|extra|turn|absolute|feed|attention|phase', Model) & grepl('200|weighted|spline', Model))
data_net_base_vs_netth = dplyr::filter(data, !grepl('additional|extra|turn|absolute|feed|attention|spline|phase', Model) & grepl('200|weighted', Model) & LatLon.difference<20)
data_netturn_baseturn = dplyr::filter(data, !grepl('th|additional|absolute|feed|attention|extra|phase|turns', Model) & grepl('trueturn|LSTM_network_diffs_200epochs', Model) & grepl('200|weighted', Model))
data_s4_phases = dplyr::filter(data, !grepl('th|additional|absolute|feed|attention|extra|turn', Model) & grepl('phase|LSTM_network_diffs_200epochs', Model) & grepl('200|weighted', Model))
data_s5_5th = dplyr::filter(data, !grepl('additional|absolute|feed|attention|extra|turns|phase|spline|extra|turnTest|weighted', Model) & grepl('200', Model) & LatLon.difference<20)
data_s6_extra = dplyr::filter(data, !grepl('th|absolute|feed|attention|turn|phase|weighted|spline', Model) & grepl('200', Model))
data_netturnextra_netturn_vs_baseturn = dplyr::filter(data, !grepl('th|additional|absolute|feed|attention', Model) & grepl('turns', Model) & grepl('200|weighted', Model))
data_netextra_netextraadditional_netadditional_net_vs_base = dplyr::filter(data, !grepl('th|absolute|feed|attention|turns', Model) & grepl('200|weighted', Model))

grid_arrange_shared_legend <- function(plots) {
    g <- ggplotGrob(plots[[1]] + theme(legend.position="bottom"))$grobs
    legend <- g[[which(sapply(g, function(x) x$name) == "guide-box")]]
    lheight <- sum(legend$height)
    hlay <- rbind(c(1,2),
                  c(4,3))
        
    # hlay <- rbind(c(10,2,6),
    #             c(11,3,7),
    #             c(12,4,8),
    #             c(13,5,9),
    #             c(NA,1,NA))
     g1 = arrangeGrob(grobs = lapply(plots, function(x)x + theme(legend.position="none")), layout_matrix = hlay)
    #g1 = arrangeGrob(grobs = lapply(plots, function(x)x + theme(legend.position="none")))
    grid.arrange(
        g1,
        legend,
        ncol = 1,
        heights = unit.c(unit(1, "npc") - lheight, lheight))
}

model_names1 = list(
    "CNN-LSTM_network_absolute" = "CNN-LSTM (non-differential)",
    "CNN-LSTM_network_diffs_" = "CNN-LSTM (differential)",
    "feed-forward_network_absolute" = "DCN (non-differential)",
    "feed-forward_network_diffs" = "DCN (differential)",
    "LSTM_network_absolute" = "LSTM (normal)",
    "LSTM_network_diffs" = "LSTM (differential)"
)

model_names2 = list(
    "LSTM_network_diffs_200epochs" = "LSTM",
    "weighted_avg_diffs" = "AVG",
    "spline_diffs" = "SPLINE"
)

model_names3 = list(
    "LSTM_network_diffs_200epochs" = "LSTM (all trajectories)",
    "LSTM_network_diffs_trueturn_200epochs" = "LSTM (turns only)",
    "LSTM_network_diffs_trueturnTest_200epochs" = "LSTM (all training, turns test)",
    "weighted_avg_diffs_trueturn" = "AVG (turns only)"
)

model_names4 = list(
    "LSTM_network_diffs_200epochs" = "LSTM (all trajectories)",
    "LSTM_network_diffs_phase_Climb_200epochs" = "LSTM (Climb)",
    "LSTM_network_diffs_phase_Cruise_200epochs" = "LSTM (Cruise)",
    "LSTM_network_diffs_phase_Descent_200epochs" = "LSTM (Descent)",
    "LSTM_network_diffs_phase_Level_200epochs" = "LSTM (Level)",
    "LSTM_network_diffs_phaseTest_Climb_200epochs" = "LSTM (all training, Climb)",
    "LSTM_network_diffs_phaseTest_Cruise_200epochs" = "LSTM (all training, Cruise)",
    "LSTM_network_diffs_phaseTest_Descent_200epochs" = "LSTM (all training, Descent)",
    "LSTM_network_diffs_phaseTest_Level_200epochs" = "LSTM (all training, Level)",
    "weighted_avg_diffs_phase_Climb" = "AVG (Climb)",
    "weighted_avg_diffs_phase_Cruise" = "AVG (Cruise)",
    "weighted_avg_diffs_phase_Descent" = "AVG (Descent)",
    "weighted_avg_diffs_phase_Level" = "AVG (Level)"
)

model_names5 = list(
    "LSTM_network_diffs_200epochs" = "LSTM",
    "LSTM_network_diffs_5th_prediction_200epochs" = "LSTM (5th prediction)",
    "LSTM_network_diffs_trueturn_200epochs" = "LSTM (turns only)",
    "LSTM_network_diffs_5th_prediction_trueturn_200epochs" = "LSTM (5th prediciton, turns only)"
)

model_names6 = list(
    "LSTM_network_diffs_200epochs" = "LSTM",
    "LSTM_network_diffs_extra_200epochs" = "LSTM + traj.-wide",
    "LSTM_network_diffs_additional_features_200epochs" = "LSTM + extra",
    "LSTM_network_diffs_extra_additional_features_200epochs" = "LSTM + traj.-wide + extra"
)

# One plot per model. x=pred position, y=difference, color=number of points
plot_techs1 = function(data, model_names, same_limits=FALSE){
    plots = list()
    for (model in unique(data$Model)){
        data_model = data[data$Model==model,]
        p = ggplot(data_model, aes(x=as.factor(Prediction.position), y=ECEF.Difference, colour=as.factor(Number.of.points), group=as.factor(Number.of.points))) +
            geom_line() + 
            ggtitle(model_names[model]) +
            viridis::scale_color_viridis(discrete=TRUE, direction = -1) +
            geom_point() +
            xlab("") + 
            ylab("") +
            labs(colour="Trajectory length")
        if(same_limits){
            max_diff = max(data$ECEF.Difference)
            p = p+ylim(0, max_diff)
        }
        plots[[length(plots)+1]] = p
    }
    print(length(plots))
    grid_arrange_shared_legend(plots)
}

# One single impossible to read plot. x=pred position, y=difference, color=number of points, shape and line=model
plot_techs2 = function(data){
    p = ggplot(data, 
               aes(x=as.factor(Prediction.position), 
                   y=ECEF.Difference, 
                   colour=as.factor(Number.of.points), 
                   group=interaction(as.factor(Number.of.points), as.factor(Model))
               )
    ) +
        geom_line(aes(linetype=Model)) +
        geom_point(aes(shape=Model)) +
        viridis::scale_color_viridis(discrete=TRUE, direction = -1)
    p
}


plot_techs1(data_diff_abs, model_names1)
plot_techs1(data_net_base, model_names2, TRUE)
plot_techs1(data_netturn_baseturn, model_names3, TRUE)
plot_techs1(data_s4_phases, model_names4, TRUE)
plot_techs1(data_s5_5th, model_names5, TRUE)
plot_techs1(data_s6_extra, model_names6, TRUE)
