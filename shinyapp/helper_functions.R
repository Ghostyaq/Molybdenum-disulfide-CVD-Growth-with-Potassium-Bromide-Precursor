library(tidyverse)
library(stringr)

# Assumes dataset already in required format, plus Spectra correctly ordered
tidy_conversion <- function(data) {
    print(class(data))
    print(is.environment(data))
    spectra_info <- tibble(
        spectrum_id = names(data)[-1],
        timestamp = data[1, -1] |> unlist() |> as.character(),
        metadata = data[2, -1] |> unlist() |> as.character()
    )
    
    tidy_spectra <- data[-c(1, 2), ] |>
        rename(x_axis = 1) |>
        mutate(x_axis = as.numeric(x_axis)) |>
        pivot_longer(
            cols = -x_axis,
            names_to = "spectrum_id",
            values_to = "intensity"
        ) |>
        mutate(intensity = as.numeric(intensity)) |>
        left_join(spectra_info, by = "spectrum_id") |>
        select(
            spectrum_id,
            timestamp,
            metadata,
            x_axis,
            intensity
        ) |>
        arrange(as.numeric(gsub("Spectrum.", "", spectrum_id))) |>
        mutate(
            spectrum_id = factor(
                spectrum_id, 
                levels = unique(spectra_info$spectrum_id), 
                ordered = TRUE)
        )
        
    tidy_spectra
}

plot_raman_spectra <- function(data, selection) {
    data <- data |>
        filter(spectrum_id %in% selection | metadata %in% selection) |>
        group_by(spectrum_id)
    
    ggplot(data, aes(x = x_axis, y = intensity, color = spectrum_id)) +
        geom_line() + 
        labs(
            x = "Raman Shift (cm^-1)",
            y = "Intensity"
        ) + 
        theme_bw()
}

