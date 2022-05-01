import matplotlib.pyplot as plt
import numpy as np
import csv
import math
import random

# Function creates an agent to another agents social network
def add_to_net(trader_list, influencer, influencee):
    trader_list[influencer].socnet[influencee] = 1
    trader_list[influencer].connections += 1


# Function used to add an extremist at the start of a market session
def new_start_extremist(traders):
    tid_list = list(traders.keys())
    influencer = {"max_connections": 0}
    for tid in random.sample(tid_list, len(tid_list)):
        if traders[tid].connections >= influencer["max_connections"] and not traders[tid].extremist:
            influencer["max_connections"] = traders[tid].connections
            influencer["tid"] = tid

    influencer_tid = influencer["tid"]
    traders[influencer_tid].opinion = 1
    traders[influencer_tid].extremist = True
    traders[influencer_tid].obstinant = 1


# Adds n extremists to the market
def initial_extreme(trader_list, n_extremists):
    current_extremists = 0
    while current_extremists < n_extremists:
        new_start_extremist(trader_list)
        current_extremists += 1
        # Sort traders by most connected traders


# Create a fully connected social network
def fully_connected_socnet(trader_list):
    for tid1 in trader_list:
        for tid2 in trader_list:
            if tid1 != tid2:
                add_to_net(trader_list, tid1, tid2)


# Create a random socnet
def random_socnet(traders, prob=0.1):
    n_traders = len(traders)
    trader_list = random.sample(list(traders.keys()), n_traders)
    for tid1 in trader_list:
        for tid2 in trader_list:
            if tid1 != tid2 and random.random() < prob:
                add_to_net(traders, tid1, tid2)


# Create a small world socnet
def small_world(traders, neighbours=10, prob=0.5):
    n_traders = len(traders)
    trader_list = random.sample(list(traders.keys()), n_traders)
    untraversed_edges = []
    traversed_edges = []
    for tid1 in trader_list:
        while traders[tid1].connections < neighbours:
            tid2 = random.choice(trader_list)
            if tid1 != tid2 and tid2 not in traders[tid1].socnet:
                add_to_net(traders, tid1, tid2)
                untraversed_edges.append([tid1, tid2])

    for edge in untraversed_edges:
        if random.random() < prob:
            rewired = False
            while not rewired:
                new_tid = random.choice(trader_list)
                old_tid = edge[0]
                end_tid = edge[1]
                if end_tid not in traders[new_tid].socnet:
                    traders[old_tid].socnet.pop(end_tid, None)
                    traders[old_tid].connections += -1

                    traders[new_tid].socnet[end_tid] = 1
                    traders[new_tid].connections += 1

                    untraversed_edges.remove(edge)
                    traversed_edges.append([new_tid, end_tid])
                    rewired = True
        else:
            untraversed_edges.remove(edge)
            traversed_edges.append(edge)


# Create a scale free socnet
def scale_free(trader_list, n=10):
    n_traders = len(trader_list)
    shuffled_tids = random.sample(list(trader_list.keys()), n_traders)
    for i in range(n_traders):
        if i != n_traders - 1:
            current_tid = shuffled_tids[i]
            next_tid = shuffled_tids[i + 1]
            add_to_net(trader_list, current_tid, next_tid)
        else:
            current_tid = shuffled_tids[i]
            next_tid = shuffled_tids[0]
            add_to_net(trader_list, current_tid, next_tid)

    # Calculate number of degrees on the graph - those at the end of
    total_connections = n_traders

    for _ in range(n):
        shuffled_t1 = random.sample(list(trader_list.keys()), n_traders)
        shuffled_t2 = random.sample(shuffled_t1, n_traders)
        for t1 in shuffled_t1:
            new_connections = 0
            for t2 in shuffled_t2:
                prob_t2 = trader_list[t2].connections / total_connections
                if random.random() < prob_t2 and t1 != t2 and t1 not in trader_list[t2].socnet:
                    add_to_net(trader_list, t2, t1)
                    new_connections += 1
            total_connections += new_connections


# connect each trader's social network -- for opinion dynamics and give it an initial opinion too -
def socnet_connect(trader_list, topology, n_extrem, initial_extrem=True):
    def market_network(trader_list):
        network = []
        for t in trader_list:
            for influencee in trader_list[t].socnet:
                network.append((t, influencee))
        return network

    # something to explore: what if sellers only connect to sellers, and buyers to buyers?
    if topology == "fully_connected":
        fully_connected_socnet(trader_list)
    elif topology == "random":
        random_socnet(trader_list)
    elif topology == "small_world":
        small_world(trader_list)
    elif topology == "scale_free":
        scale_free(trader_list)

    net = market_network(trader_list)
    # If initial extrem then we add extremists at the beginning of the market session
    if initial_extrem:
        initial_extreme(trader_list, n_extrem)
    verbose = False
    if verbose:
        for tid in trader_list:
            print('Trader, {}, opinion={}, socnet={}'.format(tid, trader_list[tid].opinion, trader_list[tid].socnet)
                  )
    return net

# Include initial extremists in socnet.
