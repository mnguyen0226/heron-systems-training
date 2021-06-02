import numpy as np
from scipy.special import softmax

d_model = 4

def main():
    print("Running")
    # Step 1: We can think of this as 3 words
    print("Step 1: INput 3 inputs, d_model = 4")
    x =np.array([[1.0, 0.0, 1.0, 0.0],   # Input 1
                [0.0, 2.0, 0.0, 2.0],   # Input 2
                [1.0, 1.0, 1.0, 1.0]])  # Input 3
    print(x)

    # Step 2: Initialize the weight matrix Q,K,V. In the original Transformer paper, the matrix has dim = 64. Here we scare it to 3 => 3x4 matrix
    print("Step 2: weights 3 dimensions x d_model=4")
    print("w_query - Q")
    w_query =np.array([[1, 0, 1],
                    [1, 0, 0],
                    [0, 0, 1],
                    [0, 1, 1]])
    print(w_query) 

    print("w_key - K")
    w_key =np.array([[0, 0, 1],
                    [1, 1, 0],
                    [0, 1, 0],
                    [1, 1, 0]])
    print(w_key)   

    print("w_value  v")
    w_value = np.array([[0, 2, 0],
                        [0, 3, 0],
                        [1, 0, 3],
                        [1, 1, 0]])
    print(w_value)

    print("Step 3: Matrix multiplication to obtain Q,K,V")
    print("Query: x * w_query")
    Q=np.matmul(x,w_query)
    print(Q)

    print("Key: x * w_key")
    K=np.matmul(x,w_key)
    print(K)

    print("Value: x * w_value")
    V=np.matmul(x,w_value)
    print(V)

    """
    At this step, we first obtain the Q1 vector of word 1 by multiply the word by the Q matrix
    We then calcualte the score of the input by multiply the Q of each to all K values (attention to other word and to itself)
    Then we softmax it and multiply with the V value of all 3 words thus we know that the word attend to other words and itself and what value it focus on
    """
    print("Step 4: Scaled Attention Scores for all 3 words")
    k_d = 1   #square root of k_d=3 rounded down to 1 for this example
    attention_scores = (Q @ K.transpose())/k_d
    print(attention_scores)

    # The attention was then applied softmax
    print("Step 5: Scaled softmax attention_scores for each vector")
    attention_scores[0]=softmax(attention_scores[0])
    attention_scores[1]=softmax(attention_scores[1])
    attention_scores[2]=softmax(attention_scores[2])
    print(attention_scores[0])
    print(attention_scores[1])
    print(attention_scores[2])

    # finalize the attention by plugging in V
    print("Step 6: attention value obtained by score1/k_d * V")
    print(V[0])
    print(V[1])
    print(V[2])
    print("Attention 1")
    attention1=attention_scores[0].reshape(-1,1)
    attention1=attention_scores[0][0]*V[0]
    print(attention1)
    print("Attention 2")
    attention2=attention_scores[0][1]*V[1]
    print(attention2)
    print("Attention 3")
    attention3=attention_scores[0][2]*V[2]
    print(attention3)

    # sum up the result:
    print("Step7: summed the results to create the first line of the output matrix")
    attention_input1=attention1+attention2+attention3
    print(attention_input1)
    
    # done

if __name__ == "__main__":
    main()