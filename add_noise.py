import h5py
import numpy as np

scale = 0.01

pressure = h5py.File('p.h5','r')
pressure_noise = h5py.File('p_n.h5','w')
dset = pressure['Pressure']
dset.copy(dset, pressure_noise)
pressure_editing = pressure_noise['Pressure']['vector_0']
mu = 0
sigma = scale
noise_p = np.random.normal(mu, sigma, pressure_editing.size)
pressure_editing[...] += noise_p
pressure.close()
pressure_noise.close()

velocity = h5py.File('u.h5','r')
velocity_noise = h5py.File('u_n.h5','w')
dset1 = velocity['Velocity']
dset1.copy(dset1, velocity_noise)
velocity_editing = velocity_noise['Velocity']['vector_0']
size_vel = int(velocity_editing.size/2)
mean = [0,0]
cov = [[scale,0],[0,scale]]
noise_u = np.random.multivariate_normal(mean, cov, size_vel).flatten()
velocity_editing[...] += noise_u
velocity.close()
velocity_noise.close()
