traffic_dates <- function() {
  dates <- seq(as.Date("2008-01-01"), as.Date("2009-03-30"), by = "day")
  removed <- as.Date(c("2008-01-01", "2008-01-21", "2008-02-18", "2008-03-09", "2008-03-31",
                       "2008-05-26", "2008-07-04", "2008-09-01", "2008-11-11", "2008-11-17",
                       "2008-12-25", "2009-01-01", "2009-01-21", "2009-02-16", "2009-03-08"))
  dates[!dates %in% removed]
}

parse_numbers <- function(s) {
  s <- trimws(gsub("\\[|\\]", "", s))
  as.numeric(strsplit(s, " +")[[1]])
}

load_traffic <- function(raw_dir, out_file = "data/processed/traffic.rds") {
  train <- readLines(file.path(raw_dir, "PEMS_train"))
  test <- readLines(file.path(raw_dir, "PEMS_test"))
  perm <- parse_numbers(readLines(file.path(raw_dir, "randperm"))[1])
  days <- c(train, test)
  out <- vector("list", length(days))
  for (i in seq_along(days)) {
    line <- days[which(perm == i)]
    rows <- strsplit(gsub("\\[|\\]", "", line), ";")[[1]]
    out[[i]] <- sapply(rows, function(r) colSums(matrix(parse_numbers(r), nrow = 6)), USE.NAMES = FALSE)
  }
  y <- t(do.call(rbind, out))
  y[is.na(y)] <- 0
  dates <- traffic_dates()
  times <- as.POSIXct(rep(as.character(dates), each = 24), tz = "UTC") + rep(0:23, length(dates)) * 3600
  stopifnot(nrow(y) == 963, ncol(y) == 10560, length(times) == ncol(y))
  res <- list(y = y, time = times, series_id = seq_len(nrow(y)))
  dir.create(dirname(out_file), recursive = TRUE, showWarnings = FALSE)
  saveRDS(res, out_file)
  res
}