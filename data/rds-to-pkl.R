library(reticulate)

# run in RSM-MSBA-SPARK container
# getwd() # should be in the 'review' directory
files <- list.files("data", pattern = "\\.rds$", full.names = TRUE, recursive = TRUE)

# loop through all data files
files <- list.files("data", pattern = "\\.rds$", full.names = TRUE, recursive = TRUE)
for (f in files) {
    df <- readr::read_rds(f)
    print(paste0("Working on:", f))
    pkl <- r_to_py(df)
    fpy <- sub("\\.rds$", ".pkl", f)
    descr <- attr(df, "description")
    py_set_attr(pkl, "description", descr)
    py_has_attr(pkl, "description")
    pkl[["_metadata"]]$append("description")
    py_save_object(pkl, fpy)
    print(paste0("Saved:", fpy))
}
