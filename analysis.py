# Here we are writing functions which will help us visualise our code
import matplotlib.pyplot as plt
import numpy as np
import math
import csv


# Loads transactions data for a whole market session
def load_transactions(fname, n_points, end_time):
    x = np.empty(0)
    y = np.empty(0)
    sum_prices = np.zeros(n_points)
    trades = np.zeros(n_points)
    segments = end_time / n_points

    with open(fname, newline='') as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            time = math.floor(float(row[1]))
            price = float(row[2])
            index = int(time // segments)
            sum_prices[index] += price
            trades[index] += 1
        average_price = sum_prices / trades

        for i in range(n_points):
            x = np.append(x, i * segments)
            y = np.append(y, average_price[i])
    return x, y


# Loads transaction data up until a specific timepoint
def load_reduced_transactions(fname, n_points, uptill):
    x = np.empty(0)
    y = np.empty(0)
    sum_prices = np.zeros(n_points)
    trades = np.zeros(n_points)
    segments = uptill / n_points

    with open(fname, newline='') as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            time = math.floor(float(row[1]))
            if time < uptill:
                price = float(row[2])
                index = int(time // segments)
                sum_prices[index] += price
                trades[index] += 1
            average_price = sum_prices / trades

        for i in range(n_points):
            x = np.append(x, i * segments)
            y = np.append(y, average_price[i])
    return x, y


# Loads opinion data for a whole market session
def load_opinion(fname, end_time):
    x = np.empty(0)
    y = np.empty(0)
    with open(fname, newline='') as csvfile:
        reader = csv.reader(csvfile)
        not_first_row = True
        for row in reader:
            if not_first_row:
                not_first_row = False
            else:
                time = float(row[0])
                for t in range(1, len(row) - 1):
                    opinion = float(row[t])
                    x = np.append(x, time)
                    y = np.append(y, opinion)
    return x, y


# Loads the connections data for a whole market session
def load_connections(fname, hist_on=False):
    connections = np.empty(0)
    with open(fname, newline='') as csvfile:
        reader = csv.reader(csvfile)
        list_data = list(reader)
        row_length = len(list_data[1]) - 1
        for i in range(row_length):
            connection_i = int(list_data[1][i])
            connections = np.append(connections, connection_i)
        total = float(list_data[1][-1])

    if hist_on:
        plt.hist(connections, bins=[1, 5, 9, 13, 17, 21, 25, 30, 35])
        plt.show()

    return total


# Loads the average opinion of a market at each timepoint
def load_average_opinion(fname, end_time):
    x = np.empty(0)
    average_ys = np.empty(0)
    with open(fname, newline='') as csvfile:
        reader = csv.reader(csvfile)
        not_first_row = True
        for row in reader:
            if not_first_row:
                not_first_row = False
            else:
                time = float(row[0])
                x = np.append(x, time)
                y = np.empty(0)
                for t in range(1, len(row) - 1):
                    opinion = float(row[t])
                    y = np.append(y, opinion)
                average_ys = np.append(average_ys, np.mean(y))
    return x, average_ys


# Use this to plot trades of a single experiment (scatter)
def plot_scatter_price(fname, end_time):
    x = np.empty(0)
    y = np.empty(0)
    with open(fname, newline='') as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            time = float(row[1])
            price = float(row[2])
            x = np.append(x, time)
            y = np.append(y, price)

    plt.ylim(ymin=50, ymax=250)
    plt.xlim(xmin=0, xmax=1800)
    plt.plot(x, y, 'x', color='black')
    plt.show()


# Use this to plot trades of a single experiment (line)
def plot_line_average_price(fname, n_points, end_time):
    def least_squares_plot(xs, ys):
        A = np.vstack([xs, np.ones(len(xs))]).T
        m, c = np.linalg.lstsq(A, ys, rcond=None)[0]
        plt.plot(xs, m * xs + c, 'r', label='Fitted line')
        plt.legend()
        return m, c

    x, y = load_transactions(fname, n_points, end_time)
    startend_averages(y)
    plt.ylim(ymin=100, ymax=400)
    plt.plot(x, y)
    lsr = True
    if lsr:
        m, c = least_squares_plot(x, y)
        print(m, c)
    plt.show()


# Plot opinion data as scatter
def plot_opinion(fname, end_time):
    x, y = load_opinion(fname, end_time)
    plt.ylim(ymin=-1, ymax=1)
    plt.xlim(xmin=0, xmax=end_time)
    plt.scatter(x, y)
    plt.show()


# Plots the change in average opinion for a single session
def average_opinion_line(fname, end):
    x = np.empty(0)
    y = np.empty(0)
    with open(fname, newline='') as csvfile:
        reader = csv.reader(csvfile)
        not_first_row = True
        for row in reader:
            if not_first_row:
                not_first_row = False
            else:
                time = float(row[0])
                ops = np.empty(0)
                x = np.append(x, time)
                for t in range(1, len(row) - 1):
                    opinion = float(row[t])
                    ops = np.append(ops, opinion)
                average = np.mean(ops)
                y = np.append(y, average)
    plt.ylim(ymin=-1, ymax=1)
    plt.xlim(xmin=0, xmax=end)
    plt.plot(x, y, '.r-')
    plt.show()


# Shows opinion plot as a histogram
def opinion_histogram(op_data, title):
    segments = [-1, -0.8, -0.6, -0.4, -0.2, 0, 0.2, 0.4, 0.6, 0.8, 1]
    plt.hist(op_data, segments)
    plt.title(title)
    plt.show()


# Returns the start and end of market average opinion of single session
def before_after_ops(fname, hist_on=False):
    with open(fname, newline='') as csvfile:
        reader = csv.reader(csvfile)
        list_data = list(reader)
        initial_ops = np.array([float(list_data[1][i]) for i in range(1, len(list_data[1]) - 1)])
        final_ops = np.array([float(list_data[-1][i]) for i in range(1, len(list_data[-1]) - 1)])
        average_initial_op = np.mean(np.array(initial_ops))
        average_final_op = np.mean(np.array(final_ops))

        if hist_on:
            opinion_histogram(initial_ops, "Opinion distribution at the start")
            opinion_histogram(final_ops, "Opinion distribution at the end")

        print("Initial Opinion average:", np.mean(initial_ops), " Final Opinion Average: ", np.mean(final_ops))
        return average_initial_op, average_final_op


# This function works out the averages of the first and last 10% of transactions - reads only transaction files
def startend_averages(ys):
    start = np.empty(0)
    end = np.empty(0)
    for i in range(len(ys) // 20):
        start_price = float(ys[i])
        start = np.append(start, start_price)
        last_price = float(ys[-i - 1])
        end = np.append(end, last_price)

    print({"Average of first 5% of transactions": np.mean(start), "Average of last 5% of transactions": np.mean(end)})


# Here we plot the average prices, standard deviation and regression lines of n iid  experiments of the same type
# Could also have the option to output the average standard deviation for transactions
def transaction_stats(trial_name, ntrials, npoints, end_time):
    # Our function for Least Squares plot
    def least_squares_plot(xs, ys):
        A = np.vstack([xs, np.ones(len(xs))]).T
        m, c = np.linalg.lstsq(A, ys, rcond=None)[0]
        plt.plot(xs, m * xs + c, 'r', label='Fitted line', linewidth=0.75)
        plt.legend()
        return m, c

    # Loop for n times
    all_values = []
    for i in range(1, ntrials + 1):
        fname = trial_name + str(i) + "_transactions.csv"
        # Load stats
        xs, values = load_transactions(fname, npoints, end_time)
        # Append stats to our existing stats - Should have n arrays of transaction prices at this point
        all_values.append(values)

    # Need to create npoints of np arrays each containing ntrials of data
    all_trans_at_t = []
    for j in range(npoints):
        transactions_at_t = np.empty(0)
        for n in range(ntrials):
            transactions_at_t = np.append(transactions_at_t, all_values[n][j])
        all_trans_at_t.append(transactions_at_t)
    # all_trans_at_t contains npoints of elements where each element has the prices of each experiment at that time

    # Once we have these we can have 3 different y-axis array elements
    ys = np.empty(0)
    sds = np.empty(0)
    plus_ys, minus_ys = np.empty(0), np.empty(0)
    for k in range(npoints):
        sd = np.std(all_trans_at_t[k])
        sds = np.append(sds, sd)
        ys = np.append(ys, np.mean(all_trans_at_t[k]))
        plus_ys = np.append(plus_ys, np.mean(all_trans_at_t[k]) + sd)
        minus_ys = np.append(minus_ys, np.mean(all_trans_at_t[k]) - sd)

    plt.ylim(ymin=100, ymax=400)
    plt.plot(xs, ys, linewidth=2)
    plt.plot(xs, plus_ys, "g--", linewidth=0.25)
    plt.plot(xs, minus_ys, "g--", linewidth=0.25)

    m, c = least_squares_plot(xs, ys)
    print("m:", float(m), " c:", c, " sd:", np.mean(sds))
    startend_averages(ys)
    # print("Average of first 5% of transactions", np.mean(), "Average of last 5% of transactions", np.mean())
    # Need to add regression line
    plt.show()


# Returns stats of opinion data of n market sessions
def opinion_stats(trial_name, ntrials, end_time):
    # Loop for n times
    all_values = []
    before, after = np.empty(0), np.empty(0)
    for i in range(1, ntrials + 1):
        fname = trial_name + str(i) + "_data.csv"
        # Load stats
        xs, values = load_average_opinion(fname, end_time)
        # Append stats to our existing stats - Should have n arrays of transaction prices at this point
        all_values.append(values)

    npoints = 199
    # Need to create npoints of np arrays each containing ntrials of data
    all_ops_at_t = []
    for j in range(199):
        ops_at_t = np.empty(0)
        for n in range(ntrials):
            ops_at_t = np.append(ops_at_t, all_values[n][j])
        all_ops_at_t.append(ops_at_t)
    # all_trans_at_t contains npoints of elements where each element has the prices of each experiment at that time

    # Once we have these we can have 3 different y-axis array elements
    ys = np.empty(0)
    for k in range(npoints):
        ys = np.append(ys, np.mean(all_ops_at_t[k]))

    print("Initial Opinion average:", ys[0], " Final Opinion Average: ", ys[-1])
    plt.ylim(ymin=-1, ymax=1)
    plt.plot(xs, ys, linewidth=2)
    plt.show()


# Returns stats of connection data of n market sessions
def connection_stats(trial_name, ntrials):
    avg_size = np.empty(0)
    for i in range(1, ntrials + 1):
        fname = trial_name + str(i) + "_connections.csv"
        size_i = load_connections(fname)
        avg_size = np.append(avg_size, size_i)
    print("Average size of network", np.mean(avg_size))


# Print market summary of n trials
def print_stats(trial_name, ntrials, npoints, end_time):
    transaction_stats(trial_name, ntrials, npoints, end_time)
    opinion_stats(trial_name, ntrials, end_time)
    connection_stats(trial_name, ntrials)


# Returns data for experiments in a csv file
def read_ids(fname):
    with open(fname, newline='') as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            end_time = int(row[-1])
            transaction_data = row[0]
            opinion_data = row[1]
            connections_data = row[2]
            print(transaction_data)
            plot_line_average_price(transaction_data, 200, end_time)
            plot_opinion(opinion_data, end_time)
            average_opinion_line(opinion_data, end_time)
            before_after_ops(opinion_data, True)
            load_connections(connections_data, True)


# This function analysises a single set of trials of our extremist run
def extremists_avg_plot(trial_id, ntrials, npoints, tag, end_time):
    def least_squares_plot(xs, ys):
        A = np.vstack([xs, np.ones(len(xs))]).T
        m, c = np.linalg.lstsq(A, ys, rcond=None)[0]
        plt.plot(xs, m * xs + c, "b", linewidth=0.75)
        return m, c

    trials = []
    extremists = [1, 3, 5, 10]
    points = 200
    for e in extremists:
        all_values = []
        trial_full_name = trial_id + "e" + str(e) + "_" + tag + "_"
        for i in range(1, ntrials + 1):
            fname = trial_full_name + str(i) + "_transactions.csv"
            # Load stats
            xs, values = load_transactions(fname, npoints, end_time)
            # Append stats to our existing stats - Should have n arrays of transaction prices at this point
            all_values.append(values)

        # Need to create npoints of np arrays each containing ntrials of data
        all_trans_at_t = []
        for j in range(npoints):
            transactions_at_t = np.empty(0)
            for n in range(ntrials):
                transactions_at_t = np.append(transactions_at_t, all_values[n][j])
            all_trans_at_t.append(transactions_at_t)
        # all_trans_at_t contains npoints of elements where each element has the prices of each experiment at that time

        # Once we have these we can have 3 different y-axis array elements
        ys = np.empty(0)
        sd = np.empty(0)
        sds = np.empty(0)
        for k in range(npoints):
            sd = np.std(all_trans_at_t[k])
            sds = np.append(sds, sd)
            ys = np.append(ys, np.mean(all_trans_at_t[k]))

        plt.ylim(ymin=100, ymax=400)
        plt.plot(xs, ys, label=str(e), linewidth=2)
        m, c = least_squares_plot(xs, ys)
        print("m:", float(m), " c:", c, " sd:", np.mean(sds))
        startend_averages(ys)
    plt.legend()
    plt.show()


# Create function which analyse n sets of trials of our extremist run
def extremists_single_plot(trial, tag, end_time):
    extremists = [1, 3, 5, 10]
    points = 200
    for i in extremists:
        fname = trial + "e" + str(i) + "_" + tag + "_1_transactions.csv"
        print(fname)
        xs_i, ys_i = load_transactions(fname, points, end_time)
        plt.ylim(100, 400)
        plt.plot(xs_i, ys_i, label=str(i))
    plt.legend()
    plt.show
