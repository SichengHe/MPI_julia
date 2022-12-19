
using MPI
MPI.Init()

comm = MPI.COMM_WORLD
size = MPI.Comm_size(comm)
rank = MPI.Comm_rank(comm)

Nx1 = 20
Nx2 = 20

x1 = LinRange(0., 1.0, Nx1)
x2 = LinRange(0., 1.0, Nx2)

function gaga(x)

    return x[1]^2 + x[2]^2

end


N = Nx1 * Nx2
N_per_node = Integer(ceil(N/size))

# Collect data
x = zeros(Float64, 2)
y = zeros(Float64, N_per_node)
k = 1
for i in 1:Nx1
    for j in 1:Nx2

        global k

        ind = (i - 1) * Nx2 + j

        x[1] = x1[i]
        x[2] = x2[j]

        # Get the corresponding process ID
        rank_ij = (ind - 1) % size

        # Run simulation using the corresponding core
        if (rank == rank_ij)
            y_loc = gaga(x)
            y[k] = y_loc

            k += 1
        end 

    end
end

# println("rank", rank, "y", y)

# Send data back
function get_loc_to_glo(rank, k, size, Nx1, Nx2)

    # Global index of the entry (flattened)
    ind_glo = (k - 1) * size + (rank + 1)

    # Get i, j indices
    i = (ind_glo - 1) ÷ Nx2 + 1
    j = ind_glo - (i - 1) * Nx2

    return [i, j]
end 

yy = zeros(Float64, (Nx1, Nx2))

for k in 1:N_per_node

    i, j = get_loc_to_glo(rank, k, size, Nx1, Nx2)

    if i <= Nx1

        yy[i, j] = y[k]

    end

end

MPI.Reduce!(yy, MPI.SUM, 0, comm)
if rank == 0
    
    println("yy", yy)
    println("y", y)
    using PlotlyJS
    plot_ref = plot(contour(x=x1, y=x2, z=yy))
    # Makie.save("plot.png", scene1)
    savefig(plot_ref, "plot.pdf")
end
