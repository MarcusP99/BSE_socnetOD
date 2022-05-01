from BSE_socnetOD import market_session

n_trials = 50
nprsh = 50
end_time = 60.0 * 60
verbose = False
dump_all = False


# Function ran a market session
def experiment(n_trials, n_prsh, end_time, sentiment, graph, od, n_extremists, pr_new_extremists, tag):
    start_time = 0.0
    range1 = (100, 100)
    supply_schedule = [{'from': start_time, 'to': end_time, 'ranges': [range1], 'stepmode': 'fixed'}]
    range2 = (400, 400)
    demand_schedule = [{'from': start_time, 'to': end_time, 'ranges': [range2], 'stepmode': 'fixed'}]
    # new customer orders arrive at each trader approx once every order_interval seconds
    order_interval = 5
    order_sched = {'sup': supply_schedule, 'dem': demand_schedule,
                   'interval': order_interval, 'timemode': 'drip-poisson'}

    sellers_spec = [('OPRZI', n_prsh)]
    buyers_spec = [('OPRZI', n_prsh)]
    traders_spec = {'sellers': sellers_spec, 'buyers': buyers_spec}

    trader_params = [sentiment, 0.0]
    frame_rate = end_time // 200  # write a row of data at this timestep?

    # n_recorded is how many trials (i.e. market sessions) to write full-fat (BIG) data-files for
    n_trials_recorded = n_trials
    trial = 1
    while trial < (n_trials + 1):
        # create unique i.d. string for this trial
        trial_id = 'oprzi{}_t{}_{}_{}_{}_e{}_{}_{}'.format(n_prsh, end_time, graph, od, sentiment, n_extremists, tag,
                                                           trial)
        print(trial_id)
        tdump = open(trial_id + '_avg_balance.csv', 'w')
        market_session(trial_id, start_time, end_time, traders_spec, order_sched, graph, od, trader_params,
                       n_extremists, pr_new_extremists, frame_rate, tdump, dump_all, verbose)
        tdump.close()

        trial = trial + 1


# Our baseline experiment involves no opinion dynamics, we will be looking at the prices at different initial sentiments
def baseline():
    n_extremists = 0
    pr_new_extremists = 0
    tag = "baseline"
    sentiments = ["Positive", "Negative"]
    for sentiment in sentiments:
        print(sentiment)
        experiment(n_trials, nprsh, end_time, sentiment, "fully_connected", "none", n_extremists, pr_new_extremists,
                   tag)


# Fully Connected network using RA and BC
def basic_ods():
    n_extremists = 0
    pr_new_extremists = 0
    tag = "actual"
    sentiments = ["Positive", "Negative", "Random"]
    ods = ["bc", "ra"]
    for sentiment in sentiments:
        for od in ods:
            experiment(n_trials, nprsh, end_time, sentiment, "fully_connected", od, n_extremists, pr_new_extremists,
                       tag)


# Look at how OD results vary with different graphs and different graph parameters
def networks():
    n_ex = 0
    pr_new_extremists = 0
    tag = "actual"
    sentiments = ["Positive"]
    ods = ["ra"]
    graphs = ["random", "small_world", "scale_free"]
    for sentiment in sentiments:
        for od in ods:
            for graph in graphs:
                experiment(n_trials, nprsh, end_time, sentiment, graph, od, n_ex, pr_new_extremists, tag)


# Look at how using the above graphs our results will vary with different number of extremists in the market
def initial_extremists():
    n_extremists = [1, 3, 5, 10]
    pr_new_extremists = 1  # This value shouldn't matter
    ods = ["ra"]
    graphs = ["fully_connected", "random", "small_world", "scale_free"]
    tag = "actual"
    for od in ods:
        for graph in graphs:
            for n_ex in n_extremists:
                experiment(n_trials, nprsh, end_time, "Neutral", graph, od, n_ex, pr_new_extremists, tag)


# This experiment was incomplete but looking into markets were extremists enter at random points
def drip_extremists():
    n_extremists = [1, 3, 5]
    pr_new_extremists = 0.001
    ods = ["ra"]
    graphs = ["random", "small_world", "scale_free"]
    tag = "test"
    for od in ods:
        for graph in graphs:
            for n_ex in n_extremists:
                experiment(n_trials, nprsh, end_time, "None", graph, od, n_ex, pr_new_extremists, tag)


# baseline() DONE
# basic_ods() DONE
# networks() DONE
# initial_extremists() DONE
# drip_extremists() Incomplete
