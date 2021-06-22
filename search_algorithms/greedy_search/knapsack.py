# Problem: Given weights and values of n items, we need to put these items in knapsack of capacity W to get the max total value in knap sack
################################################################
# Input: 
# Items as (value, weight) pairs 
# arr[] = {{60, 10}, {100, 20}, {120, 30}} 
# Knapsack Capacity, W = 50; 

# Output: 
# Maximum possible value = 240 
# by taking items of weight 10 and 20 kg and 2/3 fraction 
# of 30 kg. Hence total price will be 60+100+(2/3)(120) = 240
################################################################
# Approach: calculate the ratio value/weight for each item and 
# sort the item on basis of this ratio. Then take the item with 
# the highest ratio and add them until we can’t add the next item 
# as a whole and at the end add the next item as much as we can. 

from typing import ChainMap


class ItemValue:
    def __init__(self, weight, value, index):
        """Initialize Value of each Item in a bag

        Parameters
        ----------
        weight: int
            weight of item
        values: int
            values of item
        index: int
            index of item
        """
        self.weight = weight
        self.value = value
        self.index = index
        self.cost = value // weight 

    def __lt__(self, other):
        """Defines the behaviour of the less_than operator

        Parameters
        ----------
        other: int
            other item's value / weight cost
        """
        return self.cost  < other.cost

class FractionalKnapSack:
    @staticmethod
    def get_max_value(weight, value, capacity):
        """Get max value of knap sack bags

        Parameters
        ----------
        weight: list
            weight list of all items
        value: list
            value list of all items
        capacity: int
            capacity of the knapsack
        """
        item_value = [] # create a list of item by having their weight, value, index 
        for i in range(len(weight)):
            # go through all weight to keep track of the weight capacity that allows
            item_value.append(ItemValue(weight=weight[i], value=value[i], index=i)) # append items with weight, value, index in the bag
        
        item_value.sort(reverse=True) # sort the bag fracktion value // weight

        total_value = 0
        for item in item_value:
            # for each item that is available which is sorted
            current_weight = int(item.weight)
            current_value = int(item.value)
            if capacity - current_weight >= 0:
                # if the capacity > current weight of the largest fraction value // weight
                capacity -= current_weight # add that item to knapsack
                total_value += current_value 
            else: # else calculate the fraction of of the other item = left over capacity / weight of the current item
                fraction = capacity / current_weight
                total_value += current_value * fraction
                capacity = int(capacity - (current_weight * fraction))
                break
        return total_value

if __name__ == "__main__":
    weight = [10,40,20,30]
    value = [60,40,100,120]
    capacity = 50

    max_value_knap_sack = FractionalKnapSack.get_max_value(weight, value, capacity)
    print(f"Max value in knapsack = {max_value_knap_sack}")