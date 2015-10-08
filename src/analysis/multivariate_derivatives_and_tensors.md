$\def\R{\mathbb{R}}$
<!--- end of header material -->


# A notation and basic discussion of tensors



## Introduction

In elementary discussions of multidimensional differentiation, typically
we assume that functions $f$ map a space of column vectors, say $\R^n$,
to another such space, say $\R^m.$ In this case, the derivative of $f$ at a
point $x$ is a linear map
$$
Df(x) : \R^n \to \R^m
$$
which is the linear of the best affine approximation to $f$ near $x.$ This
map can be represented as a matrix. The components of $f$ can be treated
individually here, the $i$th row of the matrix $Df(x)$ is the transpose of
the gradient of $f_i.$

This representation of the derivative works well for first derivatives of
vector-valued vector functions. It can be generalized to functions with
matrices as inputs or outputs by simply flattening the matrices, but this
can cause us to lose a lot of the problem structure. This issue arises not
only when dealing with inherently matrix-valued functions, but also when
handling higher-order derivatives, since the second derivative of a function
$f : \R^n \to \R^m$ is, as the derivative of a derivative, already a derivative
of a matrix.

Often this is handled in an ad-hoc fashion, where the notational conventions
aren''t always clear. The [matrix cookbook][mtcb], for example, provides
many formulas that retain the structure of the problem at the expense of 
following conventions that are not always obviously mapped back to the
elementary definition of $Df(x)$ as a linear map from $\R^n$ to $\R^m$.

One standard way of handling this problem is via Kroneker-product style
representations of tensors, which--for those familiar with the
conventions--provides a notational convention for mapping linear operators on
arrays to large 2-d matrices that operate on the flattened form of those
arrays. Some formulas in the Matrix Cookbook, though not all, follow this
convention.

However, for those not familiar with the notation this flattened approach to
tensors can be more confusing than enlightening, and it also hides the fact
that when implementing algorithms using a package such as `numpy`, we would
generally encode high-dimensional operators not as large 2d matrices but as
higher-dimensional arrays.

Our goal here is to provide a set of notational conventions, and some
examples, to help those not comfortable with tensors use higher-dimensional
generaizations of matrices.



## Our Tensor Notation

For most purposes it suffices to restrict attention to tensors of no more
than four dimensions; in order to work with higher-dimensional tensors we would
need to develop intuition rather than notation, much as one develops intuition
rather than graphing methods to understand the geometry of $\R^n$ for $n$
greater than 3.

For a one- and two-dimensional tensors $v \in \R^n$ and 
$M \in \R^{m \times n}$ we will use the usual
column-vector and rectangular matrix notations, with the usual convention
that for matrix-vector operations and when transposing vectors, $\R^n$ is
considered equivalent to $\R^{n \times 1}$.
$$
v = (v_i) = \begin{pmatrix}
    v_1 \\ v_2 \\ \vdots \\ v_n
    \end{pmatrix}, \qquad
M = (M_{ij}) = \begin{pmatrix}
    M_{11} & \dots & M_{1n} \\
    \vdots & \dots & \vdots \\
    M_{m1} & \dots & M_{mn}
    \end{pmatrix}.
$$

For three-dimensional tensors, we will visualize them as column vectors of
matrices. So for $A \in \R^{a \times b \times c}$ we can express A as
$$
A = (A_{ijk}) = \begin{pmatrix}
        \begin{pmatrix}
            A_{111} & \dots & A_{11c} \\
            \vdots & \dots & \vdots \\
            A_{1b1} & \dots & A_{1bc}
        \end{pmatrix} \\
        \vdots \\
        \begin{pmatrix}
            A_{a11} & \dots & A_{a1c} \\
            \vdots & \dots & \vdots \\
            A_{ab1} & \dots & A_{abc}
        \end{pmatrix} \\
    \end{pmatrix}.
$$
Note that here, the axis we represent as a column of matrices is the first
axis. The second two axes, given any value of the first axis, are represented
as a matrix in the usual way.

And analogously, a four-dimensional tensor $B$ can be represented as a matrix
of matrices
$$
% (matrixname, i_1, i_2, n_3, n_4)
% (        #1,  #2,  #3,  #4,  #5)
\newcommand{\troxsubtensor}[5]{
    \begin{pmatrix}
        {#1}_{{#2}{#3}11}    & \dots & {#1}_{{#2}{#3}1{#5}} \\
        \vdots               & \dots & \vdots \\
        {#1}_{{#2}{#3}{#4}1} & \dots & {#1}_{{#2}{#3}{#4}{#5}}
    \end{pmatrix}
}
B = (B_{ijkl}) = \begin{pmatrix}
    \troxsubtensor{B}{1}{1}{c}{d} & \dots & \troxsubtensor{B}{1}{b}{c}{d}  \\
    \vdots                        & \dots & \vdots \\
    \troxsubtensor{B}{a}{1}{c}{d} & \dots & \troxsubtensor{B}{a}{b}{c}{d} 
\end{pmatrix}
$$

Higher dimensional tensors could be represented in an analogous way, where for
every two dimensions we add, we create a new layer in the nexted matrix
representing our tensor. Writing this out explicitly would be very cumbersome,
but thinking of this notation in the abstract can be useful.



## Tensor operations


### Left multiplication with a vector

Analogous to how we usually learn matrix operations, let us begin by defining
the product of a tensor with a vector. For clarity and without loss of
generality let us make $T$ 3-dimensional.

Suppose $T \in \R^{n\times m \times p}$ and $v \in \R^{p}.$ We can, then,
treat $T$ as a column vector 
$$
T = \begin{pmatrix}
    T_{1..} \\ T_{2..} \\ \vdots \\ T_{n..}
    \end{pmatrix}
$$
of $m \times p$ matrices. It then makes sense to define the three-dimensional
product $Tv \in \R^{n \times m \times 1}$ to be the column vector
$$
Tv = \begin{pmatrix}
    T_{1..}v \\ T_{2..}v \\ \vdots \\ T_{n..}v
    \end{pmatrix}.
$$

Note that since we identify $\R^p$ with $\R^{p \times 1}$ as in matrix-vector
multiplication, this means tensor multiplication behaves similarly to matrix
multiplication in the sense that the 'inner' dimension $p$ drops out.


### Left multiplication with a tensor

Using the same convention of the left-most dimension dropping out, we can 
define multiplication for two tensors $A \in \R^{a \times b \dots \times c}$
and $B \in \R^{c \times d \dots \times e}$ to be the tensor in
$\R^{a \times b \dots \times d \dots \times e}$ created by taking all of the
1d vectors in $A$ where we treat the last axis as the vector axis, and dotting
them with all the 1d vectors in $B$ where we treat the first axis as the
vector axis.

So, for example, if $A \in \R^{5 \times 6 \times 7}$ and
$B \in \R^{7 \times 3}$ then we can define
$$
(AB)_{ijk} = \sum_{l=1}^{7} A_{ijl} B_{lk}.
$$

This behavior is exactly what the `numpy.dot` function does for `ndarray`s.
But for clarity, let us implement our own version of `dot`:

~~~ {#pythoncode .python .numberLines}
import numpy as np

def tensor_multiply(array0, array1):
    """
    Basic tensor multiplier for ndarrays. Equivalent to np.dot().

    Implemented by appending axes to array0 (numpy auto-prepends them to
    array1).
    
    """
    sum_axis = array0.ndim - 1
    for j in range(array1.ndim - 1):
        array0 = array0[..., np.newaxis]
    return (array0 * array1).sum(axis=sum_axis)
~~~


### Altered order of multiplication

We have now generalized the multiplication of matrices to tensors. But for
many applications, we need a more general operation than taking the product
$AB$ by summing along the last axis of $A$ and the first of $B.$

For example, consider matrix-valued functions of vectors
$$
f: \R^m \to \R^{n_1 \times n_2}, \quad g: \R^m \to \R^{n_2 \times n_3}.
$$
The derivatives of $f$ and $g$ are tensors in $\R^{n_1 \times n_2 \times m}$
and $\R^{n_2 \times n_3 \times m}.$

Now consider $h = f\cdot g,$ where $\cdot$ indicates ordinary matrix
multiplication. We might hope that the 1d product rule $Dh = Df g + f Dg$ holds
in some generalized sense, and indeed we can see that it must:
$$
(f \cdot g)_{ij} \doteq \sum_k f_{ik} g_{kj},
$$
hence by the 1d product rule,
$$
\partial_{x_l} (f \cdot g)_{ij} = \sum_k
    \left(\partial_{x_l} f_{ik} g_{kg} + f_{ik} \partial_{x_l} g_{kj} \right),
$$
but unfortunately this is not the product of the tensors $Df$ and $Dg$ that
we have just described: the last axis of $Df$ is of length $m$, corresponding
to the rows of 

Usually you can work out these issues by carefully tracking the axes and their
meanings in a problem. But for the sake of being able to write down a clear
definition of the product rule, define an operator $\cdot_{k, l}$
such that $A \cdot_{k, l}$ corresponds to
  - swaping axes on $A$ so that the $k$th axis goes last
  - swapping axes on $B$ so that the $l$th axis goes first
  - computing the product in the ordinary sense.
 
In the context of the product rule, it is useful to allow negative indices
for counting backward (so as in `numpy`, -1 indicates the last axis), and also
to follow the convention that $\cdot_{k} \doteq \cdot_{k, 1},$ since generally
only the left tensor is 'out of order' when we compute these products.

With these conventions, we can see a full version of the product rule for
higher-dimensional derivatives with vector inputs:
  - let $f$ have domain $\R^{n}$ and range $\R^{T_f}$, where $\R^{T_f}$ is
  - shorthand
    for any tensor space.
  - let $g$ have domain $\R^{n}$ and range $\R^{T_g}$, where $\R^{T_g}$ is
    shorthand for any tensor space such that products of elements in
    $\R^{T_f}$ and $\R^{T_g}$ are well-defined.

Then the derivative of the tensor product function $f \cdot g$ is given by
$$
D(f \cdot g) = Df \cdot_{-2} g + f \cdot Dg.
$$

Returning to our original example, $f \cdot g$ has range $\R^{n_1 \times n_3},$
and indeed the product rule shows us that the derivative $D(f \cdot g)$ has
range $\R^{n_1 \times n_3 \times m},$ as we would expect.


### Higher-order multiplication

Consider multiplying two matrices $A$ and $B.$ The usual matrix multiplication
sums along the last axis of $A$ and first axis of $B$, which *must* have the
same length.

But, there are at least two more simple types of matrix multiplication. If the
two matrices have the same shape, we can multiply them pointwise. This is
easily expressed in terms of tensors, although for most purposes it is better
to consider pointwise operations distinct from tensors: if we identify $A \in
\R^{n \times m}$ with $\R^{n \times m \times 1}$ and $B \in \R^{n \times m}$
with $\R^{1 \times n \times m}$, then the pointwise product $A\cdot_0B$ is
equivalent to a tensor product.

More interesting is the matrix dot product: for two $n\times m$ matrices $A$
and $B$, both $\text{trace}(A^T B) = \text{trace}(B^T A) = \text{sum}(A \cdot_0
B)$ are all representations of the dot procut $\text{vec}(A) \cdot
\text{vec}(B),$ which is just the Euclidean inner product where we view $A$ and
$B$ as vectors in $\R^{nm}.$

This operation can be generalized to tensors. Suppose that the last $k$
dimensions of $A$ agree with the first $k$ dimensions of $B.$ Then we can
define $A\cdot_{:k} B$ to be the tensor obtained by flattening these dimensions
and then taking the ordinary tensor product. Note that our previous definition
of $AB$ agrees with $A\cdot_1 B,$ and that if $A$ and $B$ both have dimension
$k$ then $A \cdot_{:k} B = \text{sum}(A \cdot_0 B)$.


#### Example: the chain rule with matrices

For a simple example, consider what happens if we compose a real-valued
matrix-input function $f: \R^{m \times n} \to $\R$ with a matrix-valued
real-input function $g: \R \to \R^{m \times n}.$ The derivative of $f$
is a tensor in $\R^{1 \times n\times m},$ whereas the derivative of $g$ is
a tensor in $\R^{n\times m \times 1}$ (both of which we could identify with an
$n \times m$ matrix as the [matrix cookbook][mtcb] does, but let us leave
them as tensors to see what happens).

Now, consider the chain rule. If $h = f \circ g,$ then we know from the usual
chain rule applied to flattened versions of $f$ and $g$ that
$$
Dh(x) = \sum_{i, j} \partial_{i, j}f(g(x)) \times (D g(x))_{ij}.
$$

This cannot be expressed in terms of typical tensors without flattening the
input, which in turn can obscure problem structure (for example, if
$f(M) = \text{det}(M),$ then treating $M$ as a vector in $\R^{nm}$ would be
decidedly unhelpful!).

But, we can read directly from the formula that
$$
Dh = Df \cdot_{:2} Dg.
$$

Note that this agrees with the special solution in the [matrix cookbook][mtcb],
which is that $D f \circ g = \text{trace}(Df^T Dg),$ where we treat the
derivatives as ordinary $n \times m$ matrices.

### More multiplications

We saw in the section on
[altering the order of multiplication][### Altered order of multiplication]
that we don not always want to multiply tensors $A$ and $B$ along the last axis
of $A$ and first of $B.$ This also true of higher-order multiplications.

In general, there could arise situations where we want to multiply along the
flattened versions of an arbitrary set of axes, in which case we might
define an operator, e.g., $\cdot_{(3,5),(4,2)}$ to indicate that we want
to flatten the 3rd and 5th axes of the left side with the 4th and 2nd on the
right.

Most cases, however, do not require so much generality. Armed with an
understanding of the cases discussed thus far, the reader ought to be able to
address the proper matching of axes as needed in applications.


## Operations in terms of matrices

At a theoretical level, we have not introduced anything here which was not
already available via before-and-after reshapes. The $k$th order product of
two tensors $A \cdot_{:k} B$ is equivalent to

  * Forming a matrix $M_L(A)$ which flattens all but the first $k$ dimensions
    into the first axis, and the last $k$ dimensions into the second.
  * Forming a matrix $M_R(B) = (M_L(B^T))^T$ by flattening the first $k$
    dimensions to the first axis and the rest to the second.
  * Taking the ordinary matrix product $\tilde C \doteq M_L(A) M_R(B),$ and
  * Reshaping $\tilde C$ into a higher dimensional tensor $C$ by undoing the
    flattening operations.

We can encapsulate this idea in numpy code as follows:

~~~ {#pythoncode .python .numberLines}
import numpy as np

def dot_k(array0, array1, k):
    """
    Basic tensor multiplier for ndarrays. Equivalent to np.dot().

    Implemented by appending axes to array0 (numpy auto-prepends them to
    array1).
    
    """
    if k == 0:
        return array0 * array1
    else:
        shape_out = list(array0.shape[:-k]) + list(array1.shape[k:])
        axis_size = np.product(array0.shape[-k:])
        matrix_out = np.dot(array0.reshape(-1, axis_size),
                            array1.reshape(axis_size, -1))
        return matrix_out.reshape(shape_out)
~~~

Of course, the fact that tensor operations provide nothing new on purely
theoretical grounds does not diminish their usefulness. As a way to preserve
the structure of a problem, both in mathematical writing and in codebases
where the natural data structure is a multidimensional array rather than
a 2d matrix, they can be quite valuable.

##

Applications

ADD LINK: derivatives, the derivative product and chain rule.


[mtcb]: http://www2.imm.dtu.dk/pubdb/views/publication_details.php?id=3274
