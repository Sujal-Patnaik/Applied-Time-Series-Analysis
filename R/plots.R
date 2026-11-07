# The four figures for the report.
#
#   Rscript R/plots.R synthetic naive daily
#
# Reads the CSVs that scripts/05_calibrate.py wrote and puts PNGs in
# results/. Nothing here recomputes a number.

library(ggplot2)

args <- commandArgs(trailingOnly = TRUE)
dataset <- if (length(args) >= 1) args[1] else "synthetic"
base <- if (length(args) >= 2) args[2] else "naive"
protocol <- if (length(args) >= 3) args[3] else "daily"

suffix <- paste(dataset, base, protocol, sep = "_")
res <- file.path("results")

target <- 0.8

# keep the same order and colour in every figure
lvl <- c("base", "cqr_pooled", "cqr_horizon", "cqr_twotail", "aci")
lab <- c("Baseline", "CQR pooled", "CQR horizon-wise", "CQR two-tail", "ACI")
cols <- c("#999999", "#1f77b4", "#d62728", "#2ca02c", "#9467bd")
names(cols) <- lab

read_res <- function(prefix) {
  f <- file.path(res, paste0(prefix, "_", suffix, ".csv"))
  if (!file.exists(f)) {
    stop(paste("missing", f, "- run scripts/05_calibrate.py first"))
  }
  d <- read.csv(f)
  if ("model" %in% names(d)) {
    d$model <- factor(d$model, levels = lvl, labels = lab)
  }
  d
}

# ---- figure 1: coverage against horizon ------------------------------
h <- read_res("coverage_by_horizon")

p1 <- ggplot(h, aes(x = horizon, y = coverage, colour = model)) +
  geom_hline(yintercept = target, linetype = "dashed", colour = "black") +
  geom_line(linewidth = 0.8) +
  geom_point(size = 1.5) +
  scale_colour_manual(values = cols) +
  scale_x_continuous(breaks = seq(0, 24, 4)) +
  labs(
    title = paste("Coverage by horizon -", dataset),
    subtitle = paste("dashed line is the nominal", target, "level"),
    x = "Horizon (hours ahead)", y = "Empirical coverage", colour = NULL
  ) +
  theme_bw() +
  theme(legend.position = "bottom")

ggsave(file.path(res, paste0("fig1_coverage_horizon_", suffix, ".png")),
  p1,
  width = 8, height = 5, dpi = 150
)

# ---- figure 2: coverage over the test week, with alpha ----------------
d <- read_res("coverage_by_day")
a <- read_res("alpha_trace")

# mean adapted level per origin, rescaled onto the coverage axis
am <- aggregate(alpha ~ origin, data = a, FUN = mean)
am$shown <- 1 - am$alpha

p2 <- ggplot() +
  geom_hline(yintercept = target, linetype = "dashed", colour = "black") +
  geom_line(data = d, aes(x = day, y = coverage, colour = model), linewidth = 0.8) +
  geom_point(data = d, aes(x = day, y = coverage, colour = model), size = 1.5) +
  geom_line(
    data = am, aes(x = origin, y = shown),
    colour = "black", linetype = "dotted", linewidth = 0.7
  ) +
  scale_colour_manual(values = cols) +
  labs(
    title = paste("Coverage across the test period -", dataset),
    subtitle = "dotted black line is 1 - alpha_t, the level ACI adapted to",
    x = if (protocol == "daily") "Test day" else "Origin block",
    y = "Empirical coverage", colour = NULL
  ) +
  theme_bw() +
  theme(legend.position = "bottom")

ggsave(file.path(res, paste0("fig2_coverage_time_", suffix, ".png")),
  p2,
  width = 8, height = 5, dpi = 150
)

# ---- figure 3: width against coverage --------------------------------
hd <- read_res("headline")

p3 <- ggplot(hd, aes(x = coverage, y = width, colour = model)) +
  geom_vline(xintercept = target, linetype = "dashed", colour = "black") +
  geom_point(size = 4) +
  geom_text(aes(label = model), vjust = -1.1, size = 3, show.legend = FALSE) +
  scale_colour_manual(values = cols) +
  labs(
    title = paste("Width against coverage -", dataset),
    subtitle = "closer to the dashed line and lower is better",
    x = "Empirical coverage", y = "Normalised interval width", colour = NULL
  ) +
  theme_bw() +
  theme(legend.position = "none")

ggsave(file.path(res, paste0("fig3_width_coverage_", suffix, ".png")),
  p3,
  width = 7, height = 5, dpi = 150
)

# ---- figure 4: one series with both intervals ------------------------
e <- read.csv(file.path(res, paste0("example_series_", suffix, ".csv")))
e <- e[1:min(nrow(e), 168), ]

p4 <- ggplot(e, aes(x = step)) +
  geom_ribbon(aes(ymin = lo_cqr, ymax = hi_cqr),
    fill = "#d62728", alpha = 0.18
  ) +
  geom_ribbon(aes(ymin = lo_base, ymax = hi_base),
    fill = "#1f77b4", alpha = 0.30
  ) +
  geom_line(aes(y = actual), colour = "black", linewidth = 0.5) +
  labs(
    title = paste("One series, before and after calibration -", dataset),
    subtitle = "blue is the raw interval, red is after horizon-wise CQR",
    x = "Hour of the test period", y = "Value"
  ) +
  theme_bw()

ggsave(file.path(res, paste0("fig4_example_series_", suffix, ".png")),
  p4,
  width = 9, height = 4.5, dpi = 150
)

cat("wrote 4 figures to results/ for", suffix, "\n")
