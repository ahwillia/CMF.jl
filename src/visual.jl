module plt
  using PyPlot
end

using PyCall
@pyimport matplotlib.gridspec as gspec

include("common.jl")

function argsort_units(W)
    N = size(W, 2)
    
    # For each feature, find the component with the largest loading
    summed_lags = sum(W, dims=1)[1, :, :]
    _, ind_k = findmax(summed_lags, dims=2)

   
    # Find peak activity for each feature on its max component
    _, ind_l = findmax(W[:, ind_k], dims=1)

    # TODO make concise
    ordering = []
    for n = 1:N
        push!(ordering, (ind_k[n][2], ind_l[n][2]))
    end

    return sortperm(ordering)
end


function plot_result(W, H; data=nothing, sort=true)

    # If data is not provided, plot the prediction
    if (data == nothing)
        data = tensor_conv(W, H)
    end

    # Reorder units by model factors
    if (sort)
        ind = argsort_units(W)
    else
        ind = 1:size(W, 2)
    end

    return pyplot_result(data[ind, :], W[:, ind, :], H)
end



"""
Plots model factors and data.
    
Parameters
----------
data : ndarray
    Matrix holding data or model prediction (num_features x num_timesteps).
W : ndarray
    Temporal motifs (num_lags x num_features x num_components)
H : ndarray
    Times that motifs occur (num_componets x num_timesteps)
tmin, tmax : int
    Subset of timesteps to plot for data and H.
outer_pad : float
    Fraction of figure space to leave for all margins.
inner_pad : float
    Fraction of figure space to leave for space between subplots
data_ax_height : float
    Fractional height of main data/prediction plot.
data_ax_width : float
    Fractional width of main data/prediction plot.
figsize : tuple
    Specifies (width, height) of figure.

Returns
-------
fig : Figure
    Figure instance.
w_ax : ndarray
    Array of matplotlib Axes plotting model motif/sequences.
h_ax : ndarray
    Array of matplotlib Axes plotting time factors.
"""
function pyplot_result(data, W, H;
                       tmin=0, tmax=-1, outer_pad=.05, inner_pad=.05,
                       data_ax_height=.7, data_ax_width=.7, figsize=(10, 6))

    # Truncate data and H to desired window.
    data = data[:, tmin:tmax]
    H = H[:, tmin:tmax]
    num_components = size(H, 1)
    
    # Layout parameters for figure.
    h_ax_height = 1 - data_ax_height
    w_ax_width = 1 - data_ax_width
    pad = inner_pad + outer_pad
    
    # Create figure and axes for plotting data.
    fig = plt.figure(figsize=figsize)
    data_ax_pos = Dict(
        "left" => w_ax_width + inner_pad * w_ax_width,
        "bottom" => outer_pad,
        "right" => 1.0 - outer_pad,
        "top" => data_ax_height - inner_pad * h_ax_height,
    )

    # DEBUG
    dut = gspec.GridSpec(1, 1, data_ax_pos...)[0]
    println(dut)
    data_ax = plt.subplot(dut)
    # DEBUG
    
    data_ax.set_xticks([])
    data_ax.set_yticks([])

    # Set up axes for visualizing model motifs.
    w_ax = []
    w_ax_pos = Dict(
        "left" => outer_pad,
        "bottom" => outer_pad,
        "right" => w_ax_width,
        "top" => data_ax_height - inner_pad * h_ax_height,
        "wspace" => inner_pad,
    )
    for gs in gspec.GridSpec(1, num_components, w_ax_pos...)
        w_ax.append(plt.subplot(gs))
    end
        
    for ax in w_ax
        ax.set_yticks([])
        ax.set_xticks([])
    end
    
    # Set up axes for visualizing motif times.
    h_ax = []
    h_ax_pos = Dict(
        "left" => w_ax_width + inner_pad * w_ax_width,
        "bottom" => data_ax_height,
        "right" => 1 - outer_pad,
        "top" => 1 - outer_pad,
        "hspace" => inner_pad,
    )
            
    for gs in gspec.GridSpec(num_components, 1, h_ax_pos...)
        h_ax.append(plt.subplot(gs))
    end 
            
    for ax in h_ax
        ax.set_yticks([])
        ax.set_xticks([])
    end

    # Plot data
    data_ax.imshow(data, aspect="auto")
        
    # Plot timing factors.
    for (ax, h) in zip(h_ax, H)
        ax.plot(h)
        ax.set_xlim([0, len(h)])
        ax.axis("off")
    end
        
    # Plot motifs.
    for (ax, w) in zip(w_ax, W.T)
        ax.imshow(w, aspect="auto")
    end

    return fig, np.array(w_ax).ravel(), np.array(h_ax).ravel(), data_ax
end

;
