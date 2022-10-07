library(torch)
library(reticulate)

time_intervals <- c(30, 60, 90, 120)


np <- import("numpy")
for(nop in 5:10){
    for(time_interval in time_intervals){
        folder_path <- paste0("./training-data",time_interval,"/",nop)
        xs_path <- paste0(folder_path, "/xs.pt")
        ys_path <- paste0(folder_path, "/ys.pt")
        extras_path <- paste0(folder_path, "/extra.pt")
        
        xs <- torch_load(xs_path)
        ys <- torch_load(ys_path)
        extras <- torch_load(extras_path)
        
        xs_numpy <- r_to_py(as.array(xs))
        ys_numpy <- r_to_py(as.array(ys))
        extras_numpy <- r_to_py(as.array(extras))
        
        np$save(paste0(folder_path, "/xs.npy"), xs_numpy)
        np$save(paste0(folder_path, "/ys.npy"), ys_numpy)
        np$save(paste0(folder_path, "/extra.npy"), extras_numpy)
         
       
        # if(is.null(all_xs)){
        #     all_xs <- xs
        #     all_ys <- ys
        # } else {
        #     all_xs <- torch_cat(c(all_xs, xs))
        #     all_ys <- torch_cat(c(all_ys, ys))
        # }
        
    }
}