#######################################
# load packages and prepare functions #
#######################################
require(pacman)
p_load(openxlsx, dplyr, plyr, afex, ordinal, broom, emmeans, reshape2)


################
# data prepare #
################

dat <- read.xlsx(file.choose(), sheet = "All")



# Factor dongxi
db <- dat
db$Participant <- factor(dat$Participant)
db$Story <- factor(dat$Story, levels = 1:5)
db$Model <- factor(dat$Model, levels = c("Baseline","Gemma 3", "Gemini 2.0 Flash"))
db$Coherence <- factor(dat$Coherence, levels = 0:5)
db$Relevance <- factor(dat$Relevance, levels = 0:5)
db$Engagement <- factor(dat$Engagement, levels = 0:5)


###############################
# Linear Mixed Model Analysis #
###############################

m.Coherence <- clmm(Coherence ~ Story * Model +(1|Participant), data = db)
m.Relevance <- clmm(Relevance ~ Story * Model +(1|Participant), data = db)
m.Engagement <- clmm(Engagement ~ Story * Model +(1|Participant), data = db)



# CLMM results
models <- list(Coherence  = m.Coherence, Relevance  = m.Relevance, Engagement = m.Engagement)

out <- list()
model_contrasts_all <- NULL  

for (dv in names(models)) {
  model <- models[[dv]]

  tmp <- data.frame(tidy(model))
  tmp$dv <- dv
  tmp$sig <- ifelse(tmp$p.value <= 0.05, "*", ifelse(tmp$p.value <= 0.1, "+", ""))
  out[[paste0(dv, "_CLMM")]] <- tmp

  em <- emmeans(model, ~ Model | Story)
  contr <- contrast(em, method = "pairwise", adjust = "none") %>%
    summary(infer = TRUE) %>%
    as.data.frame()
  contr$dv <- dv
  
  model_contrasts_all <- rbind.fill(model_contrasts_all, contr)
}



# descriptive
dims <- list(Coherence = m.Coherence, Relevance = m.Relevance,Engagement = m.Engagement)
des <- NULL

for (dv in names(dims)) {
  model <- dims[[dv]]
  tmp <- data.frame(emmeans(model, ~ Story * Model))
  tmp$dv <- dv
  raw_mean <- aggregate(as.formula(paste0(dv, " ~ Story + Model")), data = dat, FUN = mean)
  names(raw_mean)[names(raw_mean) == dv] <- "M.rating"
  tmp$Story <- as.character(tmp$Story)
  raw_mean$Story <- as.character(raw_mean$Story)
  tmp <- full_join(tmp, raw_mean, by = c("Story", "Model"))
  des <- rbind.fill(des, tmp)
}

# Calculate Preference %
models <- c("Baseline", "Gemma 3", "Gemini 2.0 Flash")
stories <- as.character(sort(unique(dat$Story)))
pref <- data.frame(Story = stories)

for (m in models) {
  pref[[paste0(m, ".pref")]] <- sapply(stories, function(s) {
    sum(dat$Story == s & dat$Preference == m, na.rm = TRUE) / sum(dat$Story == s, na.rm = TRUE)
  })
}

# Comments
pref_comment <- dat %>%
  select(Participant, Story, Preference, Comment) %>%
  distinct() %>%
  na.omit() 

out$Model_Contrasts <- model_contrasts_all
out$Descript <- des
out$Prefrence <- pref
out$Comments <- pref_comment

# Export results
write.xlsx(out, file = "CLMM_results.xlsx")

