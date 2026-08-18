#!/usr/bin/env Rscript
#
# Reproduce the analysed subset of Jones & Bergen (2025) by RUNNING THE AUTHORS'
# OWN CODE, not by reimplementing their exclusion criteria.
#
# Why this matters: the release is the pre-exclusion export (1,140 completed
# games), while the paper analyses 1,023. `v2/results/stage_a/data_inspection.md`
# §3 tried to re-derive the filter from the paper's prose and got 985 / 1,014 /
# 976 / 911 depending on how the criteria were read — i.e. re-deriving it does
# not work, and any comparison with published numbers built on a re-derivation
# would be wrong. The released .Rmd files contain the real filter.
#
# Method: knitr::purl extracts the R code from each .Rmd verbatim, and we
# execute the prefix up to and including the authors' own "No. Interrogators"
# chunk. Everything after that point is modelling and plotting and does not
# touch `games`.
#
# Two `library()` lines are dropped: janitor and lmerTest are not installed here
# and are never called on this path (no clean_names/adorn_* anywhere in the
# file; the only mixed model is `lme4::glmer` ~550 lines past the exclusions).
# Nothing else is modified.
#
# The two notebooks partition the corpus by recruitment population, via an inner
# join on the profile table: SONA/UCSD is `filter(source == 2)` and Prolific is
# `filter(source == 1, study == 3)`. That is the only substantive difference
# between them, so the analysed total is the sum of the two.
#
# Usage:
#   Rscript v2/scripts/reproduce_paper_exclusions.R [outdir]
# Default outdir: v2/results/stage_a/

suppressPackageStartupMessages({
  library(knitr)
})

args <- commandArgs(trailingOnly = TRUE)
repo <- normalizePath(file.path(dirname(sub("--file=", "", grep("--file=",
          commandArgs(FALSE), value = TRUE)[1])), "..", ".."), mustWork = FALSE)
if (is.na(repo) || !dir.exists(repo)) repo <- normalizePath(getwd())
outdir <- if (length(args) >= 1) args[1] else file.path(repo, "v2/results/stage_a")
dir.create(outdir, recursive = TRUE, showWarnings = FALSE)
outdir <- normalizePath(outdir)

src <- file.path(repo, "v2/data/sources/jones_bergen_2025")
if (!dir.exists(src)) stop("source not downloaded; run `make v2-fetch-3p`")

run_one <- function(rmd, label) {
  owd <- setwd(src); on.exit(setwd(owd))   # the .Rmd reads "data/tt_*.csv"

  tmp <- tempfile(fileext = ".R")
  knitr::purl(rmd, output = tmp, quiet = TRUE, documentation = 0)
  code <- readLines(tmp)

  drop <- grepl("^\\s*library\\((janitor|lmerTest)\\)", code)
  code <- code[!drop]

  stop_at <- grep("length\\(unique\\(games\\$interrogator_user_id\\)\\)", code)
  if (!length(stop_at)) stop("could not locate the interrogator-count chunk in ", rmd)
  code <- code[seq_len(stop_at[1])]

  env <- new.env()
  invisible(capture.output(suppressWarnings(suppressMessages(
    eval(parse(text = paste(code, collapse = "\n")), envir = env)
  ))))

  g <- get("games", envir = env)
  list(
    label            = label,
    rmd              = rmd,
    lines_executed   = length(code),
    libraries_dropped = sum(drop),
    games_after_join = nrow(get("games.all", envir = env)),
    games_analysed   = nrow(g),
    interrogators_in_analysed_games = length(unique(g$interrogator_user_id)),
    profile_rows_after_prior_knowledge = nrow(get("profile", envir = env)),
    ppts_excluded_expt_aware = get("n.ppt.expt.aware", envir = env),
    games_excluded_under_2_messages = length(get("no_interaction_games", envir = env)),
    game_ids = sort(g$id)
  )
}

sona <- run_one("3p_sona_preregistered_analyses_clean.Rmd", "sona_ucsd")
prol <- run_one("3p_prolific_preregistered_analyses_clean.Rmd", "prolific")

total_games <- sona$games_analysed + prol$games_analysed
total_interrogators <- sona$interrogators_in_analysed_games +
                       prol$interrogators_in_analysed_games
total_profiles <- sona$profile_rows_after_prior_knowledge +
                  prol$profile_rows_after_prior_knowledge

fmt <- function(x) sprintf(
  "  %-9s join=%4d  analysed=%4d  interrogators=%3d  profiles_after_prior_knowledge=%3d  (excluded: %d aware ppts, %d <2-msg games)",
  x$label, x$games_after_join, x$games_analysed,
  x$interrogators_in_analysed_games, x$profile_rows_after_prior_knowledge,
  x$ppts_excluded_expt_aware, x$games_excluded_under_2_messages)

cat("Reproduction of Jones & Bergen (2025) analysed subset, by running the released .Rmd\n")
cat(fmt(sona), "\n"); cat(fmt(prol), "\n")
cat(sprintf("  TOTAL     join=%4d  analysed=%4d  interrogators=%3d  profiles=%3d\n",
            sona$games_after_join + prol$games_after_join, total_games,
            total_interrogators, total_profiles))
cat(sprintf("  paper     analysed=1023  participants=284\n"))
cat(sprintf("  MATCH games: %s | MATCH participants (profile basis): %s\n",
            total_games == 1023, total_profiles == 284))

overlap <- length(intersect(sona$game_ids, prol$game_ids))
cat(sprintf("  populations disjoint: %s (overlap=%d)\n", overlap == 0, overlap))

ids <- data.frame(
  game_id = c(sona$game_ids, prol$game_ids),
  population = c(rep("sona_ucsd", length(sona$game_ids)),
                 rep("prolific", length(prol$game_ids)))
)
ids <- ids[order(ids$game_id), ]
write.csv(ids, file.path(outdir, "paper_analysed_games.csv"), row.names = FALSE)

summary_json <- list(
  generated_by = "v2/scripts/reproduce_paper_exclusions.R",
  method = paste("knitr::purl of the released .Rmd, executed up to the authors'",
                 "own interrogator-count chunk; janitor/lmerTest library lines",
                 "dropped (uninstalled and never called on this path)"),
  paper = list(games = 1023, participants = 284),
  reproduced = list(games = total_games,
                    interrogators_in_analysed_games = total_interrogators,
                    profiles_after_prior_knowledge = total_profiles),
  by_population = list(
    sona_ucsd = sona[setdiff(names(sona), "game_ids")],
    prolific  = prol[setdiff(names(prol), "game_ids")]
  ),
  game_ids_written_to = "paper_analysed_games.csv"
)
writeLines(jsonlite::toJSON(summary_json, auto_unbox = TRUE, pretty = TRUE),
           file.path(outdir, "paper_exclusions_reproduction.json"))

cat(sprintf("\nwrote %s/paper_analysed_games.csv (%d rows)\n", outdir, nrow(ids)))
cat(sprintf("wrote %s/paper_exclusions_reproduction.json\n", outdir))

if (total_games != 1023) quit(status = 1)
