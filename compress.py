import gzip
import pickle

# Compress final_df.pkl
with open('final_df.pkl', 'rb') as f_in:
    with gzip.open('final_df.pkl.gz', 'wb') as f_out:
        pickle.dump(pickle.load(f_in), f_out)

# Compress similarity.pkl
with open('similarity.pkl', 'rb') as f_in:
    with gzip.open('similarity.pkl.gz', 'wb') as f_out:
        pickle.dump(pickle.load(f_in), f_out)
