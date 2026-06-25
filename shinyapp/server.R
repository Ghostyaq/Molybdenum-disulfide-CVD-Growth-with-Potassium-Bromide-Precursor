library(shiny)
library(DT)
library(ggplot2)
library(plotly)
library(dplyr)
library(scales)
library(shinyWidgets)
library(tidyverse)
library(shinythemes)
library(gridExtra)

# To-Do List
# How to get proper text labels in ggplotly (paste doesn't work)
# Dynamic Resizing of Raman Spectra Individual Images
# Implement All Excluding...
# Implement All
# 
# 

# Define server logic
function(input, output, session) {
    data_dir <- "../data/default Raman Spectra/"
    
    #Ready to use Dataframes
    data <- reactiveVal(
        tidy_conversion(read_csv(paste0(data_dir, "TotalTrainingData.csv")))
        )
    
    observe({
        req(data())
        #updateVirtualSelect(session, "LAS_include_individual_file", choices = unique(data()$metadata))
        updateVirtualSelect(session, "spectra_include_individual_raman", choices = unique(data()$metadata))
        updateVirtualSelect(session, "spectra_exclude_individual_raman", choices = unique(data()$metadata))
    })
    
    observeEvent(c(
        input$spectra_include_individual_raman, 
        input$spectra_exclude_individual_raman, 
        input$spectra_selection_style), {
        
        include <- input$spectra_include_individual_raman
        exclude <- input$spectra_include_individual_raman
        style   <- input$spectra_selection_style
        
        if (style == "All"){
            selected <- data()$spectrum_id
        } else if (style == "Individual Selection"){
            selected <- include
        } else {
            all_metadata <- unique(data()$metadata)
            selected <- all_metadata[!all_metadata %in% exclude]
        }
        
        plot_raman_spectra(data(), selected)
    })
}
