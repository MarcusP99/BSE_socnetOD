import random


# Given the sentiment of a market - assigns traders an initial opinion value
def generate_trader_params(sentiment, obstinant):
    if sentiment == "None":
        o_t = 0.0
    elif sentiment == "Positive":
        o_t = random.uniform(0, 1)
    elif sentiment == "Negative":
        o_t = random.uniform(0, -1)
    elif sentiment == "Neutral":
        o_t = random.uniform(-1, 1)

    return {'opinion_t0': o_t, 'obstinant': obstinant}


# Given a set of traders, chooses two traders at random
def choose_two_traders(trader_list):
    n_traders = len(trader_list)
    id1 = list(trader_list.keys())[random.randint(0, n_traders - 1)]
    id2 = id1
    while id2 == id1:
        id2 = list(trader_list.keys())[random.randint(0, n_traders - 1)]
    return id1, id2


# Chooses a random connection in the whole social network
def choose_connection(connections):
    connection = random.choice(connections)
    return connection[0], connection[1]


# For our BC model we need confidence factor, delta, intensity of interactions, weight of connection
def bounded_confidence(traders, tid1, tid2, cf=0.25, d=0.25):
    o1 = traders[tid1].opinion
    o2 = traders[tid2].opinion
    if abs(o1 - o2) < d:
        if tid1 in traders[tid2].socnet:
            op_l1 = o1 * cf + o2 * (1 - cf)
        else:
            op_l1 = o1
        if tid2 in traders[tid1].socnet:
            op_l2 = o2 * cf + o1 * (1 - cf)
        else:
            op_l2 = o2
        return op_l1, op_l2
    else:
        return o1, o2


# Check if this works in practice - this code is shit
def relative_agreement(traders, tid1, tid2, w=0.25):
    o1, o2 = traders[tid1].opinion, traders[tid2].opinion
    u1, u2 = traders[tid1].uncertainty, traders[tid2].uncertainty
    h_ij = min((o1 + u1), (o2 + u2)) - max((o1 - u1), (o2 - u2))
    h_ji = min((o2 + u2), (o1 + u1)) - max((o2 - u2), (o1 - u1))

    if h_ji > u2 and tid1 in traders[tid2].socnet:
        ra_ji = (h_ji / u2) - 1
        o1 = o1 + (w * ra_ji * (o2 - o1))
        u1 = u1 + (w * ra_ji * (u2 - u1))

    if h_ij > u1 and tid2 in traders[tid1].socnet:
        ra_ij = h_ij
        o2 = o2 + (w * ra_ij * (o1 - o2))
        u2 = u2 + (w * ra_ij * (u1 - u2))

    return o1, o2, u1, u2


# Function to add extremists in the middle of a market session - needs work with regards to opinion_t0
def new_drip_extremist(traders):
    tid_list = list(traders.keys())
    influencer = {"max_connections": 0}
    for tid in random.sample(tid_list, len(tid_list)):
        if traders[tid].connections >= influencer["max_connections"] and not traders[tid].extremist:
            influencer["max_connections"] = traders[tid].connections
            influencer["tid"] = tid

    influencer_tid = influencer["tid"]
    traders[influencer_tid].extremist = True
    traders[influencer_tid].obstinant = 1
    traders[influencer_tid].opinion_t0 = 1


# If we went for markets with traders becoming extremists during the market session and
def update_extremists(extreme_traders, direction):
    def increment(op, inc, dir):
        op = op + inc * dir
        return min(max(op, -1), 1)

    for t in extreme_traders:
        # Update extreme opinion values of traders
        if extreme_traders[t].extremist and extreme_traders[t].opinion != 1:
            extreme_traders[t].opinion = increment(extreme_traders[t].opinion, 0.001, direction)


# Updates opinion locally
def update_local(trader_list, network, od):
    tid1, tid2 = choose_connection(network)
    if od == "bc":
        op1, op2 = bounded_confidence(trader_list, tid1, tid2)
    elif od == "ra":
        updates = relative_agreement(trader_list, tid1, tid2)
        op1, op2 = updates[0], updates[1]
        trader_list[tid1].uncertainty, trader_list[tid2].uncertainty = updates[2], updates[3]

    new_op_1, new_op_2 = min((max(op1, -1.0)), +1.0), min((max(op2, -1.0)), +1.0)
    obs_1, obs_2 = trader_list[tid1].obstinant, trader_list[tid2].obstinant
    ot0_1, ot0_2 = trader_list[tid1].opinion_t0, trader_list[tid2].opinion_t0

    trader_list[tid1].opinion = (obs_1 * ot0_1) + ((1.0 - obs_1) * new_op_1)
    trader_list[tid2].opinion = (obs_2 * ot0_2) + ((1.0 - obs_2) * new_op_2)


# Function that initiates an OD interaction between a pair of traders
def opinion_dynamics(pr_local_activity, trader_list, od, network, extremists, pr_new_extremists, verbose):
    # opinion_dynamics is called on each iteration of the market_session main loop when there is an order issued
    # (i.e. up to n_traders times per second)
    # this frequency makes sense for global opinion (because LOB could change multiple times per second)
    # but for local opinion a realistic rate of interaction (among humans at least) would be slower than that
    # so pr_activity is probability of *any* local OD activity per call

    # Prefer the idea of a max number of extremists
    direction = 1

    # local opinion dynamics - extremists slowly obtaining extreme values
    if random.random() < pr_local_activity:
        update_local(trader_list, network, od)
        update_extremists(trader_list, direction)

    # Extremists gradually being entered in market
    if random.random() < pr_new_extremists and extremists > 0:
        new_drip_extremist(trader_list)
        extremists += -1

    return extremists
    # end of opinion_dynamics()
