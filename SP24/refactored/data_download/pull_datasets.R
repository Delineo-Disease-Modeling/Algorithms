#########
# Data Automation

library(censusapi)
library(data.table)
library(tidyverse)
library(tidycensus)
library(tigris)
library(sf)
library(lehdr)
library(usmap)

## tries to download a file with 1 hr timeout, retries 3 times
try_download <- function(src, dst) {
    options(timeout=3600)
    retries <- 3
    status <- 1
    while ((status != 0) && (retries > 0)) {
        status <- tryCatch({
            download.file(src, dst, mode = "wb")
        }, warning=function(e){-1}, error=function(e){-1})
        retries <- retries - 1
    }
    if (status != 0) {
        cat("download failed: ",src,"\n")
        quit()
    }
    return(status)
}

# PULL CENSUS DATA -------------
# This function will accept a vector of state codes and then pull the relevant data using the Censusapi library
# Note, you will need to provide a key

pull_census_data <- function(state_fips, year_ACS, year_DEC, ACS_table_codes, DEC_table_codes, key) {
 
    ACS_metadata <- listCensusMetadata(
        name = "acs/acs5",
        vintage = year_ACS,
        type = "variables"
    )

    DEC_metadata <- listCensusMetadata(
        name = "dec/sf1",
        vintage = year_DEC,
        type = "variables"
    )

    for(state_i in state_fips) {
        message("state = ", state_i)

        # Table Codes
        table_codes <- data.frame(table_codes = paste0("group(", ACS_table_codes, ")"),
                                  table_name = paste0("ACSDT5Y", year_ACS, ".", ACS_table_codes, "-Data"))
        
        for(row_i in 1:nrow(table_codes)) {
            message("downloading ", table_codes[row_i, "table_name"])
          
            # Make the API call and get the data
            data <- getCensus(
                name = "acs/acs5",
                vintage = year_ACS,
                vars = c(table_codes[row_i, "table_codes"]),
                region = "block group:*",
                regionin = paste0("state:", state_i, " county:*"),
                key = key
            )
            message("got ", dim_desc(data))
          
            data <- data %>%
                select(-c("state", "county", "tract", "block_group")) %>%
                select(tail(names(.), 2), everything()) %>%
                mutate_all(as.character)  
          
            data_labels <- ACS_metadata %>%
                filter(name %in% colnames(data)) %>%
                select(name, label)
          
            # Create a named vector with column labels
            label_row <- setNames(as.character(data_labels$label), data_labels$name)
            all_labels <- setNames(colnames(data), colnames(data))
            all_labels[names(label_row)] <- label_row
          
            # Convert the named vector to a data frame with one row
            label_df <- as.data.frame(t(all_labels), stringsAsFactors = FALSE)
            data_with_labels <- bind_rows(label_df, data) %>%
                select(-ends_with("EA"), -ends_with("M"), -ends_with("MA"))
          
            # Set destination to census/{state} directory where pull_datasets.R is located
            destination_folder <- file.path(dirname("./SP24/refactored/"), "census", toupper(fips_info(state_i)$abbr))

            # Create the destination folder if it doesn't exist
            if (!dir.exists(destination_folder)) {
                dir.create(destination_folder, recursive = TRUE)
            }
          
            file_name <- paste0("/", table_codes[row_i, "table_name"], ".csv")
          
            message("writing ", paste0(destination_folder, file_name))
            write.table(data_with_labels, file = paste0(destination_folder, file_name), 
                        row.names = FALSE, col.names = TRUE, sep = ",")
        }
        
        rm(row_i, table_codes)

        # DECENNIAL Data
        table_codes <- data.frame(table_codes = paste0("group(", DEC_table_codes, ")"),
                                  table_name = paste0("DECENNIALSF1", year_DEC, ".", DEC_table_codes, "-Data"))
        
        for(row_i in 1:nrow(table_codes)) {
            message("downloading ", table_codes[row_i, "table_name"])
          
            # Make the API call and get the data
            data <- getCensus(
                name = "dec/sf1",
                vintage = year_DEC,
                vars = c(table_codes[row_i, "table_codes"]),
                region = "block group:*",
                regionin = paste0("state:", state_i, " county:*"),
                key = key
            )
            message("got ", dim_desc(data))
          
            data <- data %>%
                select(-c("state", "county", "tract", "block_group")) %>%
                select(tail(names(.), 2), everything()) %>%
                mutate_all(as.character)  
          
            data_labels <- DEC_metadata %>%
                filter(name %in% colnames(data)) %>%
                select(name, label)
          
            # Create a named vector with column labels
            label_row <- setNames(as.character(data_labels$label), data_labels$name)
            all_labels <- setNames(colnames(data), colnames(data))
            all_labels[names(label_row)] <- label_row
          
            # Convert the named vector to a data frame with one row
            label_df <- as.data.frame(t(all_labels), stringsAsFactors = FALSE)
            data_with_labels <- bind_rows(label_df, data) %>%
                select(-ends_with("ERR"))
          
            file_name <- paste0("/", table_codes[row_i, "table_name"], ".csv")
          
            message("writing ", paste0(destination_folder, file_name))
            write.table(data_with_labels, file = paste0(destination_folder, file_name), 
                        row.names = FALSE, col.names = TRUE, sep = ",")
        }
    }
}
