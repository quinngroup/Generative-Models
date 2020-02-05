# synthetic dataset for testing VtPVAE_D
# will make more complex later

def make_random_dataset(num_points=20, num_dimensions=2):
    return torch.randn((num_points, num_dimensions))
