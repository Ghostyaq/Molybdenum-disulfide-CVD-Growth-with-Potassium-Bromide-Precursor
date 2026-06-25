library(shiny)
library(DT)
library(ggplot2)
library(plotly)
library(scales)
library(shinyWidgets)
library(shinythemes)
library(bslib)
library(shinycssloaders)

source("server.R")
source("helper_functions.R")

ui <- navbarPage(
    title = "MoS2 Growth",
    theme = bs_theme(
        version = 5,
        preset = "flatly"
    ),
    collapsible = TRUE,
    header = tagList(
        tags$link(rel = "stylesheet", type = "text/css", href = "styles.css"),
        tags$head(tags$script(HTML("
            function resizeFrame() {
                if (window.frameElement) {
                    var activeTab = document.querySelector('.tab-pane.active');
                    var navbar = document.querySelector('.navbar');
                    var navHeight = navbar ? navbar.offsetHeight : 0;
                    var h = activeTab ? activeTab.scrollHeight + navHeight + 20
                    : document.body.scrollHeight;
                    
                    window.frameElement.style.height = h + 'px';
                    window.frameElement.style.overflow = 'hidden';
                }
                document.documentElement.style.overflow = 'hidden';
                document.body.style.overflow = 'hidden';
            }

            // On tab click
            document.addEventListener('click', function(e) {
                var tab = e.target.closest('a[data-bs-toggle=\"tab\"], a[data-toggle=\"tab\"]');
                if (tab) {
                    setTimeout(resizeFrame, 400);
                }
            });
            
            // When any Shiny output finishes rendering, resize
            $(document).on('shiny:value shiny:outputinvalidated', function() {
                setTimeout(resizeFrame, 300);
            });
            
            // Watch for image/plot loads specifically
            $(document).on('shiny:idle', function() {
                setTimeout(resizeFrame, 300);
            });
            
            setTimeout(resizeFrame, 2000);"
        ))),
    ),
    tabPanel("Large Area Scan",
             sidebarLayout(
                 sidebarPanel(
                     radioButtons("LAS_selection_style", "Select Large Area Scan Selection Style", choices = c("Individual Selection", "All")),
                     conditionalPanel(
                         condition = "input.LAS_selection_style == 'Individual Selection'",
                         pickerInput("LAS_include_individual_file", "Raman Spectra", choices = NULL, multiple = TRUE)
                     ),
                 ),
                 mainPanel(
                     
                 )
             ),
    ),
    tabPanel(#------------------------- Raman Spectra --------------------------
             title = "Raman Spectra",
             div(class = "container-fluid",
                 div(class = "row",
                     div(class = "col-12 col-lg-3",
                         div(
                             style = "background-color: #f8f9fa; padding: 15px; 
                            border-radius: 5px; min-height: 100%;",
                             virtualSelectInput(
                                 "spectra_display_style", 
                                 "Select Display Formatting", 
                                 choices = c("Individual", "Layered")
                                 ),
                             radioButtons(
                                 "spectra_selection_style", 
                                 "Select Input Style", 
                                 choices = c(
                                     "Individual Selection", 
                                     "All Excluding...", 
                                     "All")
                                 ),
                             conditionalPanel(
                                 condition = "input.spectra_selection_style == 'Individual Selection'",
                                 virtualSelectInput(
                                     "spectra_include_individual_raman", 
                                     "Raman Spectra", 
                                     choices = NULL, multiple = TRUE, search = TRUE)
                             ),
                             conditionalPanel(
                                 condition = "input.spectra_selection_style == 'All Excluding...'",
                                 virtualSelectInput(
                                     "spectra_exclude_individual_raman", 
                                     "Raman Spectra", 
                                     choices = NULL, multiple = TRUE, search = TRUE)
                             ),
                             radioButtons(
                                 "spectra_range_style",
                                 "Range Input Style", 
                                 choices = c("All", "Custom")
                                 ),
                             conditionalPanel(
                                 condition = "input.spectra_range_style == 'Custom'",
                                 numericInput(
                                     "spectra_min", 
                                     "Raman Spectra", 
                                     value = 375, min = 0, max = 1600)
                                 ),
                             conditionalPanel(
                                 condition = "input.spectra_range_style == 'Custom'",
                                 numericInput(
                                     "spectra_max", "Raman Spectra", 
                                     value = 425, min = 0, max = 1600)
                                 ),
                             )
                        ),
                        div(class = "col-12 col-lg-9",
                            div(class = "row",
                                div(class = "col-lg-6",
                                    card(
                                        class = "graph-card",
                                        card_header("Raman Spectra Output"),
                                        plotlyOutput("spectra_raman_output") |> 
                                          withSpinner()
                                     )
                                 ),
                             )
                         )
                     )
                 )
             )
    )