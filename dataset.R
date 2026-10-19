library(torch)

tcn_dataset <- dataset(
  name = "tcn_dataset",
  initialize = function(y, cal, w, input_len, output_len, scaler = NULL) {
    self$y <- y
    self$cal <- cal
    self$w <- w
    self$input_len <- input_len
    self$output_len <- output_len
    self$scaler <- scaler
  },
  .getbatch = function(index) {
    w <- self$w[index, , drop = FALSE]
    L <- self$input_len
    H <- self$output_len
    x <- input_matrix(self$y, w, L)
    if (!is.null(self$scaler)) {
      x <- sweep(x, 2, self$scaler$mean)
      x <- sweep(x, 2, self$scaler$sd, "/")
    }
    yt <- target_matrix(self$y, w, H)
    tt <- outer(w[, "t0"], (-(L - 1)):H, "+")
    n_cal <- ncol(self$cal)
    cats <- array(0L, c(nrow(w), L + H, n_cal + 1))
    cats[, , 1] <- w[, "series"]
    for (j in seq_len(n_cal)) cats[, , j + 1] <- self$cal[tt, j]
    list(
      x = torch_tensor(x, dtype = torch_float()),
      cat_hist = torch_tensor(cats[, 1:L, , drop = FALSE], dtype = torch_long()),
      cat_fut = torch_tensor(cats[, (L + 1):(L + H), , drop = FALSE], dtype = torch_long()),
      y = torch_tensor(yt, dtype = torch_float())
    )
  },
  .length = function() {
    nrow(self$w)
  }
)
