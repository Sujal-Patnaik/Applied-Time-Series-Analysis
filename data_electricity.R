library(data.table)

load_electricity <- function(raw_file, out_file = "data/processed/electricity.rds") {
  dt <- fread(raw_file, sep = ";", dec = ",", header = TRUE)
  ids <- names(dt)[-1]
  dt[, V1 := NULL]
  x <- as.matrix(dt)
  n_hours <- nrow(x) / 4
  hourly <- rowsum(x, rep(seq_len(n_hours), each = 4))
  hourly <- hourly[8761:35064, ]
  y <- t(hourly)
  dimnames(y) <- NULL
  times <- seq(as.POSIXct("2012-01-01 00:00:00", tz = "UTC"), by = "hour", length.out = ncol(y))
  stopifnot(nrow(y) == 370, ncol(y) == 26304)
  res <- list(y = y, time = times, series_id = ids)
  dir.create(dirname(out_file), recursive = TRUE, showWarnings = FALSE)
  saveRDS(res, out_file)
  res
}
