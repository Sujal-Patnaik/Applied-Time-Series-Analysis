# Results tables as LaTeX, straight from the CSVs.
#
#   Rscript R/tables.R synthetic naive daily
#
# Every number in the report should come from here, so nothing is ever
# typed in by hand.

args <- commandArgs(trailingOnly = TRUE)
dataset <- if (length(args) >= 1) args[1] else "synthetic"
base <- if (length(args) >= 2) args[2] else "naive"
protocol <- if (length(args) >= 3) args[3] else "daily"

suffix <- paste(dataset, base, protocol, sep = "_")
res <- file.path("results")

hd <- read.csv(file.path(res, paste0("headline_", suffix, ".csv")))

lvl <- c("base", "cqr_pooled", "cqr_horizon", "cqr_twotail", "aci")
lab <- c("Baseline", "CQR pooled", "CQR horizon-wise", "CQR two-tail", "ACI")
hd$model <- factor(hd$model, levels = lvl, labels = lab)
hd <- hd[order(hd$model), ]

# ---- table 1: accuracy, which calibration never changes --------------
t1 <- data.frame(
  Model = hd$model,
  QL50 = round(hd$QL50, 4),
  QL90 = round(hd$QL90, 4),
  SMAPE = round(hd$SMAPE, 4),
  NRMSE = round(hd$NRMSE, 4),
  MASE = round(hd$MASE, 4)
)

# ---- table 2: the interval metrics, which it does --------------------
t2 <- data.frame(
  Model = hd$model,
  Coverage = round(hd$coverage, 4),
  Width = round(hd$width, 4),
  Winkler = round(hd$winkler, 4)
)

write_latex <- function(d, file, caption) {
  n <- ncol(d)
  con <- file(file, "w")
  writeLines("\\begin{table}[ht]", con)
  writeLines("\\centering", con)
  writeLines(paste0("\\caption{", caption, "}"), con)
  writeLines(
    paste0("\\begin{tabular}{l", paste(rep("r", n - 1), collapse = ""), "}"),
    con
  )
  writeLines("\\hline", con)
  writeLines(paste(paste(names(d), collapse = " & "), "\\\\"), con)
  writeLines("\\hline", con)
  for (i in 1:nrow(d)) {
    row <- paste(as.character(unlist(d[i, ])), collapse = " & ")
    writeLines(paste(row, "\\\\"), con)
  }
  writeLines("\\hline", con)
  writeLines("\\end{tabular}", con)
  writeLines("\\end{table}", con)
  close(con)
}

write_latex(
  t1, file.path(res, paste0("table_accuracy_", suffix, ".tex")),
  paste0(
    "Point accuracy on ", dataset,
    ". Identical across every row because calibration moves only the ",
    "interval ends, never the median."
  )
)

write_latex(
  t2, file.path(res, paste0("table_intervals_", suffix, ".tex")),
  paste0("Interval quality on ", dataset, ". Nominal coverage is 0.80.")
)

cat("\naccuracy\n")
print(t1, row.names = FALSE)
cat("\nintervals\n")
print(t2, row.names = FALSE)
cat("\nwrote 2 .tex files to results/\n")
