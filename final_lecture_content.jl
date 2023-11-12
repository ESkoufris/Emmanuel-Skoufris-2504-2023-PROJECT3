using MLDatasets
using Pkg
Pkg.update("MLDatasets")
train_data = MLDatasets.MNIST.traindata(Float64)

train_imgs = train_data[1]

@show typeof(train_imgs)
@show size(train_imgs)

train_labels = train_data[2]
@show typeof(train_labels);

test_data = MLDatasets.MNIST.testdata(Float64)
test_imgs = test_data[1]
test_labels = test_data[2]
@show size(test_imgs);

n_train, n_test = length(train_labels), length(test_labels)

using Plots, Measures, LaTeXStrings
println("The first 12 digits: ", train_labels[1:12])
plot([heatmap(train_data[1][:,:,k]',
            yflip=true,legend=false,c=cgrad([:black, :white])) for k in 1:12]...)

train_labels

train_data[1] |> typeof
train_data[1][:,:,1] |> vec

X = vcat([vec(train_imgs[:,:,k])' for k in 1:n_train]...)
@show size(X)
heatmap(X, legend=false)