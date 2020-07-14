from ashic.progresscb import basic_callback


def emfit(model, data, maxiter=20, tol=1e-4, callback=basic_callback, **kwargs):
    loglikelihoods = []
    converge = False
    loglikelihood = model.log_likelihood(data)  # initial loglikelihood
    i = 0
    while True:
        # E-step
        expected = model.expectation(data)
        loglikelihoods.append(loglikelihood)
        if callback:
            callback(i, model, loglikelihood, expected)
        # M-step
        model.maximization(data, expected, **kwargs)
        loglikelihood = model.log_likelihood(data)
        # check if obs likelihood decrease
        if loglikelihood < loglikelihoods[-1]:
            print "observed log-likelihood decreased at iteration {}.".format(i)
            break
        i += 1
        if abs(loglikelihood - loglikelihoods[-1]) / abs(loglikelihoods[-1]) <= tol:
            converge = True
            break
        if i >= maxiter:
            break
    expected = model.expectation(data)
    if callback:
        callback(i, model, loglikelihood, expected)
    if converge:
        message = "terminated successfully"
    elif loglikelihood < loglikelihoods[-1]:
        message = "likelihood decreased"
    elif i >= maxiter:
        message = "exceeded iteration limit"
    else:
        message = "unexpected reason"
    return model, converge, loglikelihood, expected, message
