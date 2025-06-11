import torch
from lifelines.utils import concordance_index

# Negative partial log-likelihood loss from Cox Proportional Hazards model
def cox_ph_loss(risk_scores, times, events):
    order = torch.argsort(times, descending=True)
    risk_scores = risk_scores[order]
    events = events[order]

    log_cumsum = torch.logcumsumexp(risk_scores, dim=0)
    uncensored_likelihood = risk_scores - log_cumsum
    loss = -torch.sum(uncensored_likelihood * events) / torch.sum(events)
    return loss

# Concordance index as evaluation metric
def evaluate_concordance(model, x, times, events):
    with torch.no_grad():
        risks = model(torch.tensor(x, dtype=torch.float32)).squeeze().numpy()
    return concordance_index(times, -risks, events)