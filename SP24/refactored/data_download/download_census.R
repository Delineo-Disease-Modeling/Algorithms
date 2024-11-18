library(here)
library(rjson)

config <- fromJSON(file = "./SP24/config.json")
key <- config$key # Census API key

# Read in the functions
source("./SP24/refactored/data_download/pull_datasets.R")

# which data to download from geo.json
jlist <- fromJSON(file="./SP24/refactored/data_download/geo.json")

main_fips <- unique(substr(jlist$geos, 1, 2))
county_fips <- unique(substr(jlist$geos, 3, 5))
main_abbr = fips_info(main_fips)$abbr
use_pums = if(is.null(jlist$use_pums)) main_abbr else fips_info(jlist$use_pums)$abbr
aux_abbr = if(is.null(jlist$commute_states)) vector("character") else fips_info(jlist$commute_states[!(jlist$commute_states %in% main_fips)])$abbr

# year is hardcoded for now
main_year = 2019
decennial_year = 2010

# tables needed in census.py
# B01001 - percentage of sex by age
# B09018 - Relationship to Householder for Children Under 18 Years in Households
# B09019 - Household Type (Including Living Alone) by Relationship


# acs_required = c("B01001", "B09018", "B09019", "B09020", "B09021", "B11004", "B11012", 
                # "B11016", "B19001", "B22010", "B23009", "B23025", "B25006", "B11001H", 
                # "B11001I", "C24010", "C24030")
acs_required = c("B01001", "B09019")

dec_required = c("P43")

# This function pulls the aggregated census data
# You need to provide a vector of state codes, the year, and your census API Key
# will get stored in the "census" folder, in a subfolder for each state
message("pulling census data for ",toString(main_fips))
pull_census_data(state_fips = main_fips,
                 year_ACS = main_year, 
                 year_DEC = decennial_year, 
                 ACS_table_codes = acs_required, 
                 DEC_table_codes = dec_required, 
                 key = key)