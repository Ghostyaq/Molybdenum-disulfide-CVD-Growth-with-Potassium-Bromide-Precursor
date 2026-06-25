library(tidyverse)
library(pracma)
library(data.table)
library(minpack.lm)

file_path <- "data/default LAS/300x300/Large Area Scan.csv"

raw <- read.csv(file_path, header = FALSE)
data <- raw

parse_LAS <- function(data){
    mat <- do.call(
        rbind,
        strsplit(trimws(data[,1]), "\\s+")
    )
    storage.mode(mat) <- "numeric"
    return(as.data.frame(mat))   
}

data <- parse_LAS(data)
tidy_conversion <- function(data) {
   x_axis <- data[, 1]
   data2 <- data |>
       rename(x_axis = V1) |>
       pivot_longer(
           cols = paste0("V", seq(2:length(data)) + 1),
           names_to = "id", 
           values_to = "intensity"
       ) |>
       mutate(
           id = as.numeric(sub("V", "", id)) - 1
       )
}

find_peak_locations <- function(data){
    data <- data[data$V1 > 375 & data$V1 < 420, ]
    x_axis <- data$V1
    
    e2g_idx <- which(x_axis > 375 & x_axis < 395)
    a1g_idx <- which(x_axis > 395 & x_axis < 420)
    results <- vector("list", ncol(data) - 1)
    
    for (i in 2:ncol(data)) {
        intensity <- data[[i]]
        
        peak1 <- e2g_idx[which.max(intensity[e2g_idx])]
        peak2 <- a1g_idx[which.max(intensity[a1g_idx])]
        
        results[[i-1]] <- tibble(
            id = i - 1,
            x_axis_1 = x_axis[peak1],
            intensity_1 = intensity[peak1],
            x_axis_2 = x_axis[peak2],
            intensity_2 = intensity[peak2]
        )
    }
    
    peak_locations <- bind_rows(results)
    peak_locations
}

gaussian <- function(x, A, mu, sigma) {
    A * exp(-(x - mu)^2 / (2 * sigma^2))
}

double_gaussian <- function(x, A1, mu1, sigma1, A2, mu2, sigma2, C) {
    gaussian(x, A1, mu1, sigma1) + gaussian(x, A2, mu2, sigma2) + C
}

auto_gaussian_summary <- function(data, peak_locations) {
    data <- data[data$V1 > 375 & data$V1 < 420, ]
    x <- data$V1
    results <- vector("list", nrow(peak_locations))
    
    for (i in seq_len(nrow(peak_locations))) {
        spectrum_id <- peak_locations$id[i]
        y <- data[[spectrum_id + 1]]
        
        A1_guess <- peak_locations$intensity_1[i]
        mu1_guess <- peak_locations$x_axis_1[i]
        
        A2_guess <- peak_locations$intensity_2[i]
        mu2_guess <- peak_locations$x_axis_2[i]
        
        fit <- tryCatch({
            nlsLM(y ~ double_gaussian(x, A1, mu1, sigma1, A2, mu2, sigma2, C),
                start = list(
                    A1 = A1_guess,
                    mu1 = mu1_guess,
                    sigma1 = 3,
                    
                    A2 = A2_guess,
                    mu2 = mu2_guess,
                    sigma2 = 3,
                    
                    C = min(y)
                ),
                
                lower = c(0, 375, 0.5, 0, 395, 0.5, 0),
                upper = c(Inf, 395, 15, Inf, 420, 15, Inf)
            )
        }, 
        error = function(e) {
            cat(
                "Spectrum:",
                spectrum_id,
                "\n",
                e$message,
                "\n\n"
            )
            NULL
        })
        
        if (is.null(fit)) {
            results[[i]] <- tibble(
                id = spectrum_id, mu1 = 0, mu2 = 0, fwhm1 = 0, fwhm2 = 0
                )
            next
        }
        
        p <- coef(fit)
        fwhm1 <- 2.35482 * p["sigma1"]
        fwhm2 <- 2.35482 * p["sigma2"]
        
        results[[i]] <- tibble(
            id = spectrum_id,
            mu1 = p["mu1"],
            mu2 = p["mu2"],
            fwhm1 = fwhm1,
            fwhm2 = fwhm2
            )
    }
    bind_rows(results)
}

peak_summary <- find_peak_locations(data)

gaussian_results <- auto_gaussian_summary(
    data,
    peak_summary
)

peak_summary <- peak_summary |>
    mutate(
        diff = abs(x_axis_1 - x_axis_2),
        ratio = intensity_1 / intensity_2,
        ratio = ifelse(ratio > 1, ratio, 1/ratio)
    ) |>
    merge(gaussian_results, by = "id")

heatmap_df <- peak_summary |>
    mutate(
        x = ((id - 1) %% 300) + 1,
        y = ((id - 1) %/% 300) + 1,
        diff = ifelse(intensity_1 > 730 & intensity_2 > 730, diff, 15),
        diff = ifelse(diff > 26 | diff < 18, 15, diff),
        ratio = ifelse(intensity_1 > 730 & intensity_2 > 730, ratio, 0.9),
        ratio = ifelse(ratio > 1.25, 1.25, ratio),
        mu1 = ifelse(intensity_1 > 730 & intensity_2 > 730, mu1, 0),
        mu2 = ifelse(intensity_1 > 730 & intensity_2 > 730, mu2, 0),
        fwhm1 = ifelse(intensity_1 > 730 & intensity_2 > 730, fwhm1, 0),
        fwhm2 = ifelse(intensity_1 > 730 & intensity_2 > 730, fwhm2, 0)
    ) 

cluster_num <- 8
clustering_results <- kmeans(
    select(heatmap_df, diff, mu1, mu2, fwhm1, fwhm2, ratio), 
    center = cluster_num)
ordering <- order(clustering_results$centers[, 1])
mapping <- setNames(1:cluster_num, ordering)
clustering_results$cluster <- mapping[as.character(clustering_results$cluster)]

heatmap_df$cluster <- clustering_results$cluster * 4

p <- ggplot(heatmap_df, aes(x = x, y = y, fill = cluster)) +
    geom_tile() +
    coord_equal() +
    scale_y_reverse() +
    scale_fill_gradientn(colors = c("red", "yellow", "blue")) + 
    labs(fill = "Peak Separation") + 
    theme_bw()
p
ggplotly(p)

write.csv(heatmap_df, file = "../paraview_data/analysis_results.csv", row.names = FALSE)

# ggplotly(ggplot(data, aes(x = V1, y = V41)) + geom_line() + theme_bw())