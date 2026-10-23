library(yaml)

load_config <- function(path) {
  cfg <- read_yaml(path)
  cfg$dilations <- unlist(cfg$dilations)
  cfg$quantiles <- unlist(cfg$quantiles)
  cfg$calendar <- unlist(cfg$calendar)
  cfg
}

get_device <- function() {
  if (torch::cuda_is_available()) torch::torch_device("cuda") else torch::torch_device("cpu")
}
