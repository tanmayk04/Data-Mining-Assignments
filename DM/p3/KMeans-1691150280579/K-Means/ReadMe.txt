How to Run?


Make sure that the three datasets (pendigit, satellite and yeast) are present in the same folder as the kmeans.py program. To run the program, please use the command:

python kmeans.py <training dataset>

For example, to check the yeast dataset, you can run:
python kmeans.py yeast_training.txt


The program takes more time to complete its execution based on how large the sample file is. For example, program runs faster for yeast dataset, while it takes more time for pendigit dataset. 

The overall SSE must decrease as the number of clusters increase, which is shown by the plot.