from utils import generate_contexutal_docs


# = = = = = = = = = = = = = = =

path_to_data = 'data/'
filename = 'documents_final.npy'
window_size = 6  # should be even

# = = = = = = = = = = = = = = =

generate_contexutal_docs(path_to_data, filename, window_size)