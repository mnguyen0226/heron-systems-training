    # TODO; Run the greedy search in here - Minh
    # For army in results[0:25]:
    #   For unit in units: - don't need this
    #       army.append(unit) - don't need this 
    #   win_prob = net.forward(army)
    #   wins.append(win_prob)
    # top_win = wins.argmax() -> Position
    # print(army)

# ---------------------
# 1/ Randomly generated 100 armies using 70% army budget
# 2/ Select the top 25 of those armies based on win prob
# 3/ Do 1,2,3 level search on those  armies with remaining 30% budget => Complicated
    # army = []
    # Get current armies from evaluation above (append)
    # with the remainder, do the greedy search => karthik
    # append greedy search - army => get full army
    # Run evaluation
    # append army-eval to armies-eval pair
# Sort armies by eval
# Return the top army 

# ---------------------
# 4/ This results in 75 total armies at the end
# 5/ Return the top army based on the win_prop from the list of 75

# Beam search is a heuristic search algo that explore
# graph by expanding the most promsing node in a limited sear

# Best-first search is a graph search which orders all partoal solution
# accoding to some heuristic.

# In beam search, only predetermines number of best
# partial solutions are kept as candidates

from build.lib.gamebreaker.search.base import Search


Q: Is the 25 still valid?

Greedy Search
A* Search