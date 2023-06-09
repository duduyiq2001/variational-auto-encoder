��
��
^
AssignVariableOp
resource
value"dtype"
dtypetype"
validate_shapebool( �
�
BiasAdd

value"T	
bias"T
output"T""
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
8
Const
output"dtype"
valuetensor"
dtypetype
�
Conv2DBackpropInput
input_sizes
filter"T
out_backprop"T
output"T"
Ttype:	
2"
strides	list(int)"
use_cudnn_on_gpubool(",
paddingstring:
SAMEVALIDEXPLICIT""
explicit_paddings	list(int)
 "-
data_formatstringNHWC:
NHWCNCHW" 
	dilations	list(int)

$
DisableCopyOnRead
resource�
�
FusedBatchNormV3
x"T

scale"U
offset"U	
mean"U
variance"U
y"T

batch_mean"U
batch_variance"U
reserve_space_1"U
reserve_space_2"U
reserve_space_3"U"
Ttype:
2"
Utype:
2"
epsilonfloat%��8"&
exponential_avg_factorfloat%  �?";
data_formatstringNHWC:
NHWCNCHWNDHWCNCDHW"
is_trainingbool(
.
Identity

input"T
output"T"	
Ttype
\
	LeakyRelu
features"T
activations"T"
alphafloat%��L>"
Ttype0:
2
u
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:
2	
�
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool("
allow_missing_filesbool( �

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
@
ReadVariableOp
resource
value"dtype"
dtypetype�
[
Reshape
tensor"T
shape"Tshape
output"T"	
Ttype"
Tshapetype0:
2	
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0�
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0�
?
Select
	condition

t"T
e"T
output"T"	
Ttype
d
Shape

input"T&
output"out_type��out_type"	
Ttype"
out_typetype0:
2	
H
ShardedFilename
basename	
shard

num_shards
filename
0
Sigmoid
x"T
y"T"
Ttype:

2
�
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring ��
@
StaticRegexFullMatch	
input

output
"
patternstring
�
StridedSlice

input"T
begin"Index
end"Index
strides"Index
output"T"	
Ttype"
Indextype:
2	"

begin_maskint "
end_maskint "
ellipsis_maskint "
new_axis_maskint "
shrink_axis_maskint 
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 
�
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 �"serve*2.12.02v2.12.0-rc1-12-g0db597d0d758�
�
convTranspose2D6/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*&
shared_nameconvTranspose2D6/bias
{
)convTranspose2D6/bias/Read/ReadVariableOpReadVariableOpconvTranspose2D6/bias*
_output_shapes
:*
dtype0
�
convTranspose2D6/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: *(
shared_nameconvTranspose2D6/kernel
�
+convTranspose2D6/kernel/Read/ReadVariableOpReadVariableOpconvTranspose2D6/kernel*&
_output_shapes
: *
dtype0
~
bn5/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape: *$
shared_namebn5/moving_variance
w
'bn5/moving_variance/Read/ReadVariableOpReadVariableOpbn5/moving_variance*
_output_shapes
: *
dtype0
v
bn5/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape: * 
shared_namebn5/moving_mean
o
#bn5/moving_mean/Read/ReadVariableOpReadVariableOpbn5/moving_mean*
_output_shapes
: *
dtype0
h
bn5/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name
bn5/beta
a
bn5/beta/Read/ReadVariableOpReadVariableOpbn5/beta*
_output_shapes
: *
dtype0
j
	bn5/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	bn5/gamma
c
bn5/gamma/Read/ReadVariableOpReadVariableOp	bn5/gamma*
_output_shapes
: *
dtype0
�
convTranspose2D5/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *&
shared_nameconvTranspose2D5/bias
{
)convTranspose2D5/bias/Read/ReadVariableOpReadVariableOpconvTranspose2D5/bias*
_output_shapes
: *
dtype0
�
convTranspose2D5/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: @*(
shared_nameconvTranspose2D5/kernel
�
+convTranspose2D5/kernel/Read/ReadVariableOpReadVariableOpconvTranspose2D5/kernel*&
_output_shapes
: @*
dtype0
~
bn4/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*$
shared_namebn4/moving_variance
w
'bn4/moving_variance/Read/ReadVariableOpReadVariableOpbn4/moving_variance*
_output_shapes
:@*
dtype0
v
bn4/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:@* 
shared_namebn4/moving_mean
o
#bn4/moving_mean/Read/ReadVariableOpReadVariableOpbn4/moving_mean*
_output_shapes
:@*
dtype0
h
bn4/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*
shared_name
bn4/beta
a
bn4/beta/Read/ReadVariableOpReadVariableOpbn4/beta*
_output_shapes
:@*
dtype0
j
	bn4/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*
shared_name	bn4/gamma
c
bn4/gamma/Read/ReadVariableOpReadVariableOp	bn4/gamma*
_output_shapes
:@*
dtype0
�
convTranspose2D4/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*&
shared_nameconvTranspose2D4/bias
{
)convTranspose2D4/bias/Read/ReadVariableOpReadVariableOpconvTranspose2D4/bias*
_output_shapes
:@*
dtype0
�
convTranspose2D4/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@@*(
shared_nameconvTranspose2D4/kernel
�
+convTranspose2D4/kernel/Read/ReadVariableOpReadVariableOpconvTranspose2D4/kernel*&
_output_shapes
:@@*
dtype0
~
bn3/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*$
shared_namebn3/moving_variance
w
'bn3/moving_variance/Read/ReadVariableOpReadVariableOpbn3/moving_variance*
_output_shapes
:@*
dtype0
v
bn3/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:@* 
shared_namebn3/moving_mean
o
#bn3/moving_mean/Read/ReadVariableOpReadVariableOpbn3/moving_mean*
_output_shapes
:@*
dtype0
h
bn3/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*
shared_name
bn3/beta
a
bn3/beta/Read/ReadVariableOpReadVariableOpbn3/beta*
_output_shapes
:@*
dtype0
j
	bn3/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*
shared_name	bn3/gamma
c
bn3/gamma/Read/ReadVariableOpReadVariableOp	bn3/gamma*
_output_shapes
:@*
dtype0
�
convTranspose2D3/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*&
shared_nameconvTranspose2D3/bias
{
)convTranspose2D3/bias/Read/ReadVariableOpReadVariableOpconvTranspose2D3/bias*
_output_shapes
:@*
dtype0
�
convTranspose2D3/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@@*(
shared_nameconvTranspose2D3/kernel
�
+convTranspose2D3/kernel/Read/ReadVariableOpReadVariableOpconvTranspose2D3/kernel*&
_output_shapes
:@@*
dtype0
~
bn2/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*$
shared_namebn2/moving_variance
w
'bn2/moving_variance/Read/ReadVariableOpReadVariableOpbn2/moving_variance*
_output_shapes
:@*
dtype0
v
bn2/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:@* 
shared_namebn2/moving_mean
o
#bn2/moving_mean/Read/ReadVariableOpReadVariableOpbn2/moving_mean*
_output_shapes
:@*
dtype0
h
bn2/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*
shared_name
bn2/beta
a
bn2/beta/Read/ReadVariableOpReadVariableOpbn2/beta*
_output_shapes
:@*
dtype0
j
	bn2/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*
shared_name	bn2/gamma
c
bn2/gamma/Read/ReadVariableOpReadVariableOp	bn2/gamma*
_output_shapes
:@*
dtype0
�
convTranspose2D2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*&
shared_nameconvTranspose2D2/bias
{
)convTranspose2D2/bias/Read/ReadVariableOpReadVariableOpconvTranspose2D2/bias*
_output_shapes
:@*
dtype0
�
convTranspose2D2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@ *(
shared_nameconvTranspose2D2/kernel
�
+convTranspose2D2/kernel/Read/ReadVariableOpReadVariableOpconvTranspose2D2/kernel*&
_output_shapes
:@ *
dtype0
~
bn1/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape: *$
shared_namebn1/moving_variance
w
'bn1/moving_variance/Read/ReadVariableOpReadVariableOpbn1/moving_variance*
_output_shapes
: *
dtype0
v
bn1/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape: * 
shared_namebn1/moving_mean
o
#bn1/moving_mean/Read/ReadVariableOpReadVariableOpbn1/moving_mean*
_output_shapes
: *
dtype0
h
bn1/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name
bn1/beta
a
bn1/beta/Read/ReadVariableOpReadVariableOpbn1/beta*
_output_shapes
: *
dtype0
j
	bn1/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	bn1/gamma
c
bn1/gamma/Read/ReadVariableOpReadVariableOp	bn1/gamma*
_output_shapes
: *
dtype0
�
convTranspose2D1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *&
shared_nameconvTranspose2D1/bias
{
)convTranspose2D1/bias/Read/ReadVariableOpReadVariableOpconvTranspose2D1/bias*
_output_shapes
: *
dtype0
�
convTranspose2D1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:  *(
shared_nameconvTranspose2D1/kernel
�
+convTranspose2D1/kernel/Read/ReadVariableOpReadVariableOpconvTranspose2D1/kernel*&
_output_shapes
:  *
dtype0
o
dense1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*
shared_namedense1/bias
h
dense1/bias/Read/ReadVariableOpReadVariableOpdense1/bias*
_output_shapes	
:�*
dtype0
w
dense1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	
�*
shared_namedense1/kernel
p
!dense1/kernel/Read/ReadVariableOpReadVariableOpdense1/kernel*
_output_shapes
:	
�*
dtype0
}
serving_default_inputLayerPlaceholder*'
_output_shapes
:���������
*
dtype0*
shape:���������

�
StatefulPartitionedCallStatefulPartitionedCallserving_default_inputLayerdense1/kerneldense1/biasconvTranspose2D1/kernelconvTranspose2D1/bias	bn1/gammabn1/betabn1/moving_meanbn1/moving_varianceconvTranspose2D2/kernelconvTranspose2D2/bias	bn2/gammabn2/betabn2/moving_meanbn2/moving_varianceconvTranspose2D3/kernelconvTranspose2D3/bias	bn3/gammabn3/betabn3/moving_meanbn3/moving_varianceconvTranspose2D4/kernelconvTranspose2D4/bias	bn4/gammabn4/betabn4/moving_meanbn4/moving_varianceconvTranspose2D5/kernelconvTranspose2D5/bias	bn5/gammabn5/betabn5/moving_meanbn5/moving_varianceconvTranspose2D6/kernelconvTranspose2D6/bias*.
Tin'
%2#*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������*D
_read_only_resource_inputs&
$"	
 !"*-
config_proto

CPU

GPU 2J 8� *,
f'R%
#__inference_signature_wrapper_16278

NoOpNoOp
�}
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*�}
value�}B�} B�}
�
layer-0
layer_with_weights-0
layer-1
layer-2
layer_with_weights-1
layer-3
layer_with_weights-2
layer-4
layer-5
layer_with_weights-3
layer-6
layer_with_weights-4
layer-7
	layer-8

layer_with_weights-5

layer-9
layer_with_weights-6
layer-10
layer-11
layer_with_weights-7
layer-12
layer_with_weights-8
layer-13
layer-14
layer_with_weights-9
layer-15
layer_with_weights-10
layer-16
layer-17
layer_with_weights-11
layer-18
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
_default_save_signature

signatures*
* 
�
	variables
trainable_variables
regularization_losses
	keras_api
 __call__
*!&call_and_return_all_conditional_losses

"kernel
#bias*
�
$	variables
%trainable_variables
&regularization_losses
'	keras_api
(__call__
*)&call_and_return_all_conditional_losses* 
�
*	variables
+trainable_variables
,regularization_losses
-	keras_api
.__call__
*/&call_and_return_all_conditional_losses

0kernel
1bias
 2_jit_compiled_convolution_op*
�
3	variables
4trainable_variables
5regularization_losses
6	keras_api
7__call__
*8&call_and_return_all_conditional_losses
9axis
	:gamma
;beta
<moving_mean
=moving_variance*
�
>	variables
?trainable_variables
@regularization_losses
A	keras_api
B__call__
*C&call_and_return_all_conditional_losses* 
�
D	variables
Etrainable_variables
Fregularization_losses
G	keras_api
H__call__
*I&call_and_return_all_conditional_losses

Jkernel
Kbias
 L_jit_compiled_convolution_op*
�
M	variables
Ntrainable_variables
Oregularization_losses
P	keras_api
Q__call__
*R&call_and_return_all_conditional_losses
Saxis
	Tgamma
Ubeta
Vmoving_mean
Wmoving_variance*
�
X	variables
Ytrainable_variables
Zregularization_losses
[	keras_api
\__call__
*]&call_and_return_all_conditional_losses* 
�
^	variables
_trainable_variables
`regularization_losses
a	keras_api
b__call__
*c&call_and_return_all_conditional_losses

dkernel
ebias
 f_jit_compiled_convolution_op*
�
g	variables
htrainable_variables
iregularization_losses
j	keras_api
k__call__
*l&call_and_return_all_conditional_losses
maxis
	ngamma
obeta
pmoving_mean
qmoving_variance*
�
r	variables
strainable_variables
tregularization_losses
u	keras_api
v__call__
*w&call_and_return_all_conditional_losses* 
�
x	variables
ytrainable_variables
zregularization_losses
{	keras_api
|__call__
*}&call_and_return_all_conditional_losses

~kernel
bias
!�_jit_compiled_convolution_op*
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
	�axis

�gamma
	�beta
�moving_mean
�moving_variance*
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses* 
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�kernel
	�bias
!�_jit_compiled_convolution_op*
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
	�axis

�gamma
	�beta
�moving_mean
�moving_variance*
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses* 
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�kernel
	�bias
!�_jit_compiled_convolution_op*
�
"0
#1
02
13
:4
;5
<6
=7
J8
K9
T10
U11
V12
W13
d14
e15
n16
o17
p18
q19
~20
21
�22
�23
�24
�25
�26
�27
�28
�29
�30
�31
�32
�33*
�
"0
#1
02
13
:4
;5
J6
K7
T8
U9
d10
e11
n12
o13
~14
15
�16
�17
�18
�19
�20
�21
�22
�23*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
	variables
trainable_variables
regularization_losses
__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*
:
�trace_0
�trace_1
�trace_2
�trace_3* 
:
�trace_0
�trace_1
�trace_2
�trace_3* 
* 

�serving_default* 

"0
#1*

"0
#1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
	variables
trainable_variables
regularization_losses
 __call__
*!&call_and_return_all_conditional_losses
&!"call_and_return_conditional_losses*

�trace_0* 

�trace_0* 
]W
VARIABLE_VALUEdense1/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE*
YS
VARIABLE_VALUEdense1/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
$	variables
%trainable_variables
&regularization_losses
(__call__
*)&call_and_return_all_conditional_losses
&)"call_and_return_conditional_losses* 

�trace_0* 

�trace_0* 

00
11*

00
11*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
*	variables
+trainable_variables
,regularization_losses
.__call__
*/&call_and_return_all_conditional_losses
&/"call_and_return_conditional_losses*

�trace_0* 

�trace_0* 
ga
VARIABLE_VALUEconvTranspose2D1/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE*
c]
VARIABLE_VALUEconvTranspose2D1/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 
 
:0
;1
<2
=3*

:0
;1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
3	variables
4trainable_variables
5regularization_losses
7__call__
*8&call_and_return_all_conditional_losses
&8"call_and_return_conditional_losses*

�trace_0
�trace_1* 

�trace_0
�trace_1* 
* 
XR
VARIABLE_VALUE	bn1/gamma5layer_with_weights-2/gamma/.ATTRIBUTES/VARIABLE_VALUE*
VP
VARIABLE_VALUEbn1/beta4layer_with_weights-2/beta/.ATTRIBUTES/VARIABLE_VALUE*
d^
VARIABLE_VALUEbn1/moving_mean;layer_with_weights-2/moving_mean/.ATTRIBUTES/VARIABLE_VALUE*
lf
VARIABLE_VALUEbn1/moving_variance?layer_with_weights-2/moving_variance/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
>	variables
?trainable_variables
@regularization_losses
B__call__
*C&call_and_return_all_conditional_losses
&C"call_and_return_conditional_losses* 

�trace_0* 

�trace_0* 

J0
K1*

J0
K1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
D	variables
Etrainable_variables
Fregularization_losses
H__call__
*I&call_and_return_all_conditional_losses
&I"call_and_return_conditional_losses*

�trace_0* 

�trace_0* 
ga
VARIABLE_VALUEconvTranspose2D2/kernel6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUE*
c]
VARIABLE_VALUEconvTranspose2D2/bias4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 
 
T0
U1
V2
W3*

T0
U1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
M	variables
Ntrainable_variables
Oregularization_losses
Q__call__
*R&call_and_return_all_conditional_losses
&R"call_and_return_conditional_losses*

�trace_0
�trace_1* 

�trace_0
�trace_1* 
* 
XR
VARIABLE_VALUE	bn2/gamma5layer_with_weights-4/gamma/.ATTRIBUTES/VARIABLE_VALUE*
VP
VARIABLE_VALUEbn2/beta4layer_with_weights-4/beta/.ATTRIBUTES/VARIABLE_VALUE*
d^
VARIABLE_VALUEbn2/moving_mean;layer_with_weights-4/moving_mean/.ATTRIBUTES/VARIABLE_VALUE*
lf
VARIABLE_VALUEbn2/moving_variance?layer_with_weights-4/moving_variance/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
X	variables
Ytrainable_variables
Zregularization_losses
\__call__
*]&call_and_return_all_conditional_losses
&]"call_and_return_conditional_losses* 

�trace_0* 

�trace_0* 

d0
e1*

d0
e1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
^	variables
_trainable_variables
`regularization_losses
b__call__
*c&call_and_return_all_conditional_losses
&c"call_and_return_conditional_losses*

�trace_0* 

�trace_0* 
ga
VARIABLE_VALUEconvTranspose2D3/kernel6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUE*
c]
VARIABLE_VALUEconvTranspose2D3/bias4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 
 
n0
o1
p2
q3*

n0
o1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
g	variables
htrainable_variables
iregularization_losses
k__call__
*l&call_and_return_all_conditional_losses
&l"call_and_return_conditional_losses*

�trace_0
�trace_1* 

�trace_0
�trace_1* 
* 
XR
VARIABLE_VALUE	bn3/gamma5layer_with_weights-6/gamma/.ATTRIBUTES/VARIABLE_VALUE*
VP
VARIABLE_VALUEbn3/beta4layer_with_weights-6/beta/.ATTRIBUTES/VARIABLE_VALUE*
d^
VARIABLE_VALUEbn3/moving_mean;layer_with_weights-6/moving_mean/.ATTRIBUTES/VARIABLE_VALUE*
lf
VARIABLE_VALUEbn3/moving_variance?layer_with_weights-6/moving_variance/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
r	variables
strainable_variables
tregularization_losses
v__call__
*w&call_and_return_all_conditional_losses
&w"call_and_return_conditional_losses* 

�trace_0* 

�trace_0* 

~0
1*

~0
1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
x	variables
ytrainable_variables
zregularization_losses
|__call__
*}&call_and_return_all_conditional_losses
&}"call_and_return_conditional_losses*

�trace_0* 

�trace_0* 
ga
VARIABLE_VALUEconvTranspose2D4/kernel6layer_with_weights-7/kernel/.ATTRIBUTES/VARIABLE_VALUE*
c]
VARIABLE_VALUEconvTranspose2D4/bias4layer_with_weights-7/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 
$
�0
�1
�2
�3*

�0
�1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses*

�trace_0
�trace_1* 

�trace_0
�trace_1* 
* 
XR
VARIABLE_VALUE	bn4/gamma5layer_with_weights-8/gamma/.ATTRIBUTES/VARIABLE_VALUE*
VP
VARIABLE_VALUEbn4/beta4layer_with_weights-8/beta/.ATTRIBUTES/VARIABLE_VALUE*
d^
VARIABLE_VALUEbn4/moving_mean;layer_with_weights-8/moving_mean/.ATTRIBUTES/VARIABLE_VALUE*
lf
VARIABLE_VALUEbn4/moving_variance?layer_with_weights-8/moving_variance/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses* 

�trace_0* 

�trace_0* 

�0
�1*

�0
�1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses*

�trace_0* 

�trace_0* 
ga
VARIABLE_VALUEconvTranspose2D5/kernel6layer_with_weights-9/kernel/.ATTRIBUTES/VARIABLE_VALUE*
c]
VARIABLE_VALUEconvTranspose2D5/bias4layer_with_weights-9/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 
$
�0
�1
�2
�3*

�0
�1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses*

�trace_0
�trace_1* 

�trace_0
�trace_1* 
* 
YS
VARIABLE_VALUE	bn5/gamma6layer_with_weights-10/gamma/.ATTRIBUTES/VARIABLE_VALUE*
WQ
VARIABLE_VALUEbn5/beta5layer_with_weights-10/beta/.ATTRIBUTES/VARIABLE_VALUE*
e_
VARIABLE_VALUEbn5/moving_mean<layer_with_weights-10/moving_mean/.ATTRIBUTES/VARIABLE_VALUE*
mg
VARIABLE_VALUEbn5/moving_variance@layer_with_weights-10/moving_variance/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses* 

�trace_0* 

�trace_0* 

�0
�1*

�0
�1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses*

�trace_0* 

�trace_0* 
hb
VARIABLE_VALUEconvTranspose2D6/kernel7layer_with_weights-11/kernel/.ATTRIBUTES/VARIABLE_VALUE*
d^
VARIABLE_VALUEconvTranspose2D6/bias5layer_with_weights-11/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 
N
<0
=1
V2
W3
p4
q5
�6
�7
�8
�9*
�
0
1
2
3
4
5
6
7
	8

9
10
11
12
13
14
15
16
17
18*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 

<0
=1*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 

V0
W1*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 

p0
q1*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 

�0
�1*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 

�0
�1*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
�
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filenamedense1/kerneldense1/biasconvTranspose2D1/kernelconvTranspose2D1/bias	bn1/gammabn1/betabn1/moving_meanbn1/moving_varianceconvTranspose2D2/kernelconvTranspose2D2/bias	bn2/gammabn2/betabn2/moving_meanbn2/moving_varianceconvTranspose2D3/kernelconvTranspose2D3/bias	bn3/gammabn3/betabn3/moving_meanbn3/moving_varianceconvTranspose2D4/kernelconvTranspose2D4/bias	bn4/gammabn4/betabn4/moving_meanbn4/moving_varianceconvTranspose2D5/kernelconvTranspose2D5/bias	bn5/gammabn5/betabn5/moving_meanbn5/moving_varianceconvTranspose2D6/kernelconvTranspose2D6/biasConst*/
Tin(
&2$*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *'
f"R 
__inference__traced_save_17722
�
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamedense1/kerneldense1/biasconvTranspose2D1/kernelconvTranspose2D1/bias	bn1/gammabn1/betabn1/moving_meanbn1/moving_varianceconvTranspose2D2/kernelconvTranspose2D2/bias	bn2/gammabn2/betabn2/moving_meanbn2/moving_varianceconvTranspose2D3/kernelconvTranspose2D3/bias	bn3/gammabn3/betabn3/moving_meanbn3/moving_varianceconvTranspose2D4/kernelconvTranspose2D4/bias	bn4/gammabn4/betabn4/moving_meanbn4/moving_varianceconvTranspose2D5/kernelconvTranspose2D5/bias	bn5/gammabn5/betabn5/moving_meanbn5/moving_varianceconvTranspose2D6/kernelconvTranspose2D6/bias*.
Tin'
%2#*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� **
f%R#
!__inference__traced_restore_17834Į
�
�
>__inference_bn3_layer_call_and_return_conditional_losses_15141

inputs%
readvariableop_resource:@'
readvariableop_1_resource:@6
(fusedbatchnormv3_readvariableop_resource:@8
*fusedbatchnormv3_readvariableop_1_resource:@
identity��AssignNewValue�AssignNewValue_1�FusedBatchNormV3/ReadVariableOp�!FusedBatchNormV3/ReadVariableOp_1�ReadVariableOp�ReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype0�
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype0�
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype0�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+���������������������������@:@:@:@:@:*
epsilon%o�:*
exponential_avg_factor%
�#<�
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(�
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(}
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+���������������������������@�
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+���������������������������@: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+���������������������������@
 
_user_specified_nameinputs
� 
�
K__inference_convTranspose2D2_layer_call_and_return_conditional_losses_17038

inputsB
(conv2d_transpose_readvariableop_resource:@ -
biasadd_readvariableop_resource:@
identity��BiasAdd/ReadVariableOp�conv2d_transpose/ReadVariableOpI
ShapeShapeinputs*
T0*
_output_shapes
::��]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_1StridedSliceShape:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_2StridedSliceShape:output:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskG
mul/yConst*
_output_shapes
: *
dtype0*
value	B :U
mulMulstrided_slice_1:output:0mul/y:output:0*
T0*
_output_shapes
: I
mul_1/yConst*
_output_shapes
: *
dtype0*
value	B :Y
mul_1Mulstrided_slice_2:output:0mul_1/y:output:0*
T0*
_output_shapes
: I
stack/3Const*
_output_shapes
: *
dtype0*
value	B :@y
stackPackstrided_slice:output:0mul:z:0	mul_1:z:0stack/3:output:0*
N*
T0*
_output_shapes
:_
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_3StridedSlicestack:output:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask�
conv2d_transpose/ReadVariableOpReadVariableOp(conv2d_transpose_readvariableop_resource*&
_output_shapes
:@ *
dtype0�
conv2d_transposeConv2DBackpropInputstack:output:0'conv2d_transpose/ReadVariableOp:value:0inputs*
T0*A
_output_shapes/
-:+���������������������������@*
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0�
BiasAddBiasAddconv2d_transpose:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+���������������������������@y
IdentityIdentityBiasAdd:output:0^NoOp*
T0*A
_output_shapes/
-:+���������������������������@�
NoOpNoOp^BiasAdd/ReadVariableOp ^conv2d_transpose/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:+��������������������������� : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2B
conv2d_transpose/ReadVariableOpconv2d_transpose/ReadVariableOp:i e
A
_output_shapes/
-:+��������������������������� 
 
_user_specified_nameinputs
�
�
0__inference_convTranspose2D4_layer_call_fn_17233

inputs!
unknown:@@
	unknown_0:@
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+���������������������������@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *T
fORM
K__inference_convTranspose2D4_layer_call_and_return_conditional_losses_15220�
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+���������������������������@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:+���������������������������@: : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+���������������������������@
 
_user_specified_nameinputs
��
�
 __inference__wrapped_model_14862

inputlayer@
-decoder_dense1_matmul_readvariableop_resource:	
�=
.decoder_dense1_biasadd_readvariableop_resource:	�[
Adecoder_convtranspose2d1_conv2d_transpose_readvariableop_resource:  F
8decoder_convtranspose2d1_biasadd_readvariableop_resource: 1
#decoder_bn1_readvariableop_resource: 3
%decoder_bn1_readvariableop_1_resource: B
4decoder_bn1_fusedbatchnormv3_readvariableop_resource: D
6decoder_bn1_fusedbatchnormv3_readvariableop_1_resource: [
Adecoder_convtranspose2d2_conv2d_transpose_readvariableop_resource:@ F
8decoder_convtranspose2d2_biasadd_readvariableop_resource:@1
#decoder_bn2_readvariableop_resource:@3
%decoder_bn2_readvariableop_1_resource:@B
4decoder_bn2_fusedbatchnormv3_readvariableop_resource:@D
6decoder_bn2_fusedbatchnormv3_readvariableop_1_resource:@[
Adecoder_convtranspose2d3_conv2d_transpose_readvariableop_resource:@@F
8decoder_convtranspose2d3_biasadd_readvariableop_resource:@1
#decoder_bn3_readvariableop_resource:@3
%decoder_bn3_readvariableop_1_resource:@B
4decoder_bn3_fusedbatchnormv3_readvariableop_resource:@D
6decoder_bn3_fusedbatchnormv3_readvariableop_1_resource:@[
Adecoder_convtranspose2d4_conv2d_transpose_readvariableop_resource:@@F
8decoder_convtranspose2d4_biasadd_readvariableop_resource:@1
#decoder_bn4_readvariableop_resource:@3
%decoder_bn4_readvariableop_1_resource:@B
4decoder_bn4_fusedbatchnormv3_readvariableop_resource:@D
6decoder_bn4_fusedbatchnormv3_readvariableop_1_resource:@[
Adecoder_convtranspose2d5_conv2d_transpose_readvariableop_resource: @F
8decoder_convtranspose2d5_biasadd_readvariableop_resource: 1
#decoder_bn5_readvariableop_resource: 3
%decoder_bn5_readvariableop_1_resource: B
4decoder_bn5_fusedbatchnormv3_readvariableop_resource: D
6decoder_bn5_fusedbatchnormv3_readvariableop_1_resource: [
Adecoder_convtranspose2d6_conv2d_transpose_readvariableop_resource: F
8decoder_convtranspose2d6_biasadd_readvariableop_resource:
identity��+decoder/bn1/FusedBatchNormV3/ReadVariableOp�-decoder/bn1/FusedBatchNormV3/ReadVariableOp_1�decoder/bn1/ReadVariableOp�decoder/bn1/ReadVariableOp_1�+decoder/bn2/FusedBatchNormV3/ReadVariableOp�-decoder/bn2/FusedBatchNormV3/ReadVariableOp_1�decoder/bn2/ReadVariableOp�decoder/bn2/ReadVariableOp_1�+decoder/bn3/FusedBatchNormV3/ReadVariableOp�-decoder/bn3/FusedBatchNormV3/ReadVariableOp_1�decoder/bn3/ReadVariableOp�decoder/bn3/ReadVariableOp_1�+decoder/bn4/FusedBatchNormV3/ReadVariableOp�-decoder/bn4/FusedBatchNormV3/ReadVariableOp_1�decoder/bn4/ReadVariableOp�decoder/bn4/ReadVariableOp_1�+decoder/bn5/FusedBatchNormV3/ReadVariableOp�-decoder/bn5/FusedBatchNormV3/ReadVariableOp_1�decoder/bn5/ReadVariableOp�decoder/bn5/ReadVariableOp_1�/decoder/convTranspose2D1/BiasAdd/ReadVariableOp�8decoder/convTranspose2D1/conv2d_transpose/ReadVariableOp�/decoder/convTranspose2D2/BiasAdd/ReadVariableOp�8decoder/convTranspose2D2/conv2d_transpose/ReadVariableOp�/decoder/convTranspose2D3/BiasAdd/ReadVariableOp�8decoder/convTranspose2D3/conv2d_transpose/ReadVariableOp�/decoder/convTranspose2D4/BiasAdd/ReadVariableOp�8decoder/convTranspose2D4/conv2d_transpose/ReadVariableOp�/decoder/convTranspose2D5/BiasAdd/ReadVariableOp�8decoder/convTranspose2D5/conv2d_transpose/ReadVariableOp�/decoder/convTranspose2D6/BiasAdd/ReadVariableOp�8decoder/convTranspose2D6/conv2d_transpose/ReadVariableOp�%decoder/dense1/BiasAdd/ReadVariableOp�$decoder/dense1/MatMul/ReadVariableOp�
$decoder/dense1/MatMul/ReadVariableOpReadVariableOp-decoder_dense1_matmul_readvariableop_resource*
_output_shapes
:	
�*
dtype0�
decoder/dense1/MatMulMatMul
inputlayer,decoder/dense1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
%decoder/dense1/BiasAdd/ReadVariableOpReadVariableOp.decoder_dense1_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
decoder/dense1/BiasAddBiasAdddecoder/dense1/MatMul:product:0-decoder/dense1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������w
decoder/reshapeLayer/ShapeShapedecoder/dense1/BiasAdd:output:0*
T0*
_output_shapes
::��r
(decoder/reshapeLayer/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: t
*decoder/reshapeLayer/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:t
*decoder/reshapeLayer/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
"decoder/reshapeLayer/strided_sliceStridedSlice#decoder/reshapeLayer/Shape:output:01decoder/reshapeLayer/strided_slice/stack:output:03decoder/reshapeLayer/strided_slice/stack_1:output:03decoder/reshapeLayer/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
$decoder/reshapeLayer/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :f
$decoder/reshapeLayer/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :f
$decoder/reshapeLayer/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B : �
"decoder/reshapeLayer/Reshape/shapePack+decoder/reshapeLayer/strided_slice:output:0-decoder/reshapeLayer/Reshape/shape/1:output:0-decoder/reshapeLayer/Reshape/shape/2:output:0-decoder/reshapeLayer/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:�
decoder/reshapeLayer/ReshapeReshapedecoder/dense1/BiasAdd:output:0+decoder/reshapeLayer/Reshape/shape:output:0*
T0*/
_output_shapes
:��������� �
decoder/convTranspose2D1/ShapeShape%decoder/reshapeLayer/Reshape:output:0*
T0*
_output_shapes
::��v
,decoder/convTranspose2D1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: x
.decoder/convTranspose2D1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:x
.decoder/convTranspose2D1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
&decoder/convTranspose2D1/strided_sliceStridedSlice'decoder/convTranspose2D1/Shape:output:05decoder/convTranspose2D1/strided_slice/stack:output:07decoder/convTranspose2D1/strided_slice/stack_1:output:07decoder/convTranspose2D1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskb
 decoder/convTranspose2D1/stack/1Const*
_output_shapes
: *
dtype0*
value	B :b
 decoder/convTranspose2D1/stack/2Const*
_output_shapes
: *
dtype0*
value	B :b
 decoder/convTranspose2D1/stack/3Const*
_output_shapes
: *
dtype0*
value	B : �
decoder/convTranspose2D1/stackPack/decoder/convTranspose2D1/strided_slice:output:0)decoder/convTranspose2D1/stack/1:output:0)decoder/convTranspose2D1/stack/2:output:0)decoder/convTranspose2D1/stack/3:output:0*
N*
T0*
_output_shapes
:x
.decoder/convTranspose2D1/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: z
0decoder/convTranspose2D1/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:z
0decoder/convTranspose2D1/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
(decoder/convTranspose2D1/strided_slice_1StridedSlice'decoder/convTranspose2D1/stack:output:07decoder/convTranspose2D1/strided_slice_1/stack:output:09decoder/convTranspose2D1/strided_slice_1/stack_1:output:09decoder/convTranspose2D1/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask�
8decoder/convTranspose2D1/conv2d_transpose/ReadVariableOpReadVariableOpAdecoder_convtranspose2d1_conv2d_transpose_readvariableop_resource*&
_output_shapes
:  *
dtype0�
)decoder/convTranspose2D1/conv2d_transposeConv2DBackpropInput'decoder/convTranspose2D1/stack:output:0@decoder/convTranspose2D1/conv2d_transpose/ReadVariableOp:value:0%decoder/reshapeLayer/Reshape:output:0*
T0*/
_output_shapes
:��������� *
paddingSAME*
strides
�
/decoder/convTranspose2D1/BiasAdd/ReadVariableOpReadVariableOp8decoder_convtranspose2d1_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0�
 decoder/convTranspose2D1/BiasAddBiasAdd2decoder/convTranspose2D1/conv2d_transpose:output:07decoder/convTranspose2D1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:��������� z
decoder/bn1/ReadVariableOpReadVariableOp#decoder_bn1_readvariableop_resource*
_output_shapes
: *
dtype0~
decoder/bn1/ReadVariableOp_1ReadVariableOp%decoder_bn1_readvariableop_1_resource*
_output_shapes
: *
dtype0�
+decoder/bn1/FusedBatchNormV3/ReadVariableOpReadVariableOp4decoder_bn1_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype0�
-decoder/bn1/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp6decoder_bn1_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype0�
decoder/bn1/FusedBatchNormV3FusedBatchNormV3)decoder/convTranspose2D1/BiasAdd:output:0"decoder/bn1/ReadVariableOp:value:0$decoder/bn1/ReadVariableOp_1:value:03decoder/bn1/FusedBatchNormV3/ReadVariableOp:value:05decoder/bn1/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:��������� : : : : :*
epsilon%o�:*
is_training( �
decoder/lReLU1/LeakyRelu	LeakyRelu decoder/bn1/FusedBatchNormV3:y:0*/
_output_shapes
:��������� *
alpha%���>�
decoder/convTranspose2D2/ShapeShape&decoder/lReLU1/LeakyRelu:activations:0*
T0*
_output_shapes
::��v
,decoder/convTranspose2D2/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: x
.decoder/convTranspose2D2/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:x
.decoder/convTranspose2D2/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
&decoder/convTranspose2D2/strided_sliceStridedSlice'decoder/convTranspose2D2/Shape:output:05decoder/convTranspose2D2/strided_slice/stack:output:07decoder/convTranspose2D2/strided_slice/stack_1:output:07decoder/convTranspose2D2/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskb
 decoder/convTranspose2D2/stack/1Const*
_output_shapes
: *
dtype0*
value	B :b
 decoder/convTranspose2D2/stack/2Const*
_output_shapes
: *
dtype0*
value	B :b
 decoder/convTranspose2D2/stack/3Const*
_output_shapes
: *
dtype0*
value	B :@�
decoder/convTranspose2D2/stackPack/decoder/convTranspose2D2/strided_slice:output:0)decoder/convTranspose2D2/stack/1:output:0)decoder/convTranspose2D2/stack/2:output:0)decoder/convTranspose2D2/stack/3:output:0*
N*
T0*
_output_shapes
:x
.decoder/convTranspose2D2/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: z
0decoder/convTranspose2D2/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:z
0decoder/convTranspose2D2/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
(decoder/convTranspose2D2/strided_slice_1StridedSlice'decoder/convTranspose2D2/stack:output:07decoder/convTranspose2D2/strided_slice_1/stack:output:09decoder/convTranspose2D2/strided_slice_1/stack_1:output:09decoder/convTranspose2D2/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask�
8decoder/convTranspose2D2/conv2d_transpose/ReadVariableOpReadVariableOpAdecoder_convtranspose2d2_conv2d_transpose_readvariableop_resource*&
_output_shapes
:@ *
dtype0�
)decoder/convTranspose2D2/conv2d_transposeConv2DBackpropInput'decoder/convTranspose2D2/stack:output:0@decoder/convTranspose2D2/conv2d_transpose/ReadVariableOp:value:0&decoder/lReLU1/LeakyRelu:activations:0*
T0*/
_output_shapes
:���������@*
paddingSAME*
strides
�
/decoder/convTranspose2D2/BiasAdd/ReadVariableOpReadVariableOp8decoder_convtranspose2d2_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0�
 decoder/convTranspose2D2/BiasAddBiasAdd2decoder/convTranspose2D2/conv2d_transpose:output:07decoder/convTranspose2D2/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@z
decoder/bn2/ReadVariableOpReadVariableOp#decoder_bn2_readvariableop_resource*
_output_shapes
:@*
dtype0~
decoder/bn2/ReadVariableOp_1ReadVariableOp%decoder_bn2_readvariableop_1_resource*
_output_shapes
:@*
dtype0�
+decoder/bn2/FusedBatchNormV3/ReadVariableOpReadVariableOp4decoder_bn2_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype0�
-decoder/bn2/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp6decoder_bn2_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype0�
decoder/bn2/FusedBatchNormV3FusedBatchNormV3)decoder/convTranspose2D2/BiasAdd:output:0"decoder/bn2/ReadVariableOp:value:0$decoder/bn2/ReadVariableOp_1:value:03decoder/bn2/FusedBatchNormV3/ReadVariableOp:value:05decoder/bn2/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:���������@:@:@:@:@:*
epsilon%o�:*
is_training( �
decoder/lReLU2/LeakyRelu	LeakyRelu decoder/bn2/FusedBatchNormV3:y:0*/
_output_shapes
:���������@*
alpha%���>�
decoder/convTranspose2D3/ShapeShape&decoder/lReLU2/LeakyRelu:activations:0*
T0*
_output_shapes
::��v
,decoder/convTranspose2D3/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: x
.decoder/convTranspose2D3/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:x
.decoder/convTranspose2D3/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
&decoder/convTranspose2D3/strided_sliceStridedSlice'decoder/convTranspose2D3/Shape:output:05decoder/convTranspose2D3/strided_slice/stack:output:07decoder/convTranspose2D3/strided_slice/stack_1:output:07decoder/convTranspose2D3/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskb
 decoder/convTranspose2D3/stack/1Const*
_output_shapes
: *
dtype0*
value	B :b
 decoder/convTranspose2D3/stack/2Const*
_output_shapes
: *
dtype0*
value	B :b
 decoder/convTranspose2D3/stack/3Const*
_output_shapes
: *
dtype0*
value	B :@�
decoder/convTranspose2D3/stackPack/decoder/convTranspose2D3/strided_slice:output:0)decoder/convTranspose2D3/stack/1:output:0)decoder/convTranspose2D3/stack/2:output:0)decoder/convTranspose2D3/stack/3:output:0*
N*
T0*
_output_shapes
:x
.decoder/convTranspose2D3/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: z
0decoder/convTranspose2D3/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:z
0decoder/convTranspose2D3/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
(decoder/convTranspose2D3/strided_slice_1StridedSlice'decoder/convTranspose2D3/stack:output:07decoder/convTranspose2D3/strided_slice_1/stack:output:09decoder/convTranspose2D3/strided_slice_1/stack_1:output:09decoder/convTranspose2D3/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask�
8decoder/convTranspose2D3/conv2d_transpose/ReadVariableOpReadVariableOpAdecoder_convtranspose2d3_conv2d_transpose_readvariableop_resource*&
_output_shapes
:@@*
dtype0�
)decoder/convTranspose2D3/conv2d_transposeConv2DBackpropInput'decoder/convTranspose2D3/stack:output:0@decoder/convTranspose2D3/conv2d_transpose/ReadVariableOp:value:0&decoder/lReLU2/LeakyRelu:activations:0*
T0*/
_output_shapes
:���������@*
paddingSAME*
strides
�
/decoder/convTranspose2D3/BiasAdd/ReadVariableOpReadVariableOp8decoder_convtranspose2d3_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0�
 decoder/convTranspose2D3/BiasAddBiasAdd2decoder/convTranspose2D3/conv2d_transpose:output:07decoder/convTranspose2D3/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@z
decoder/bn3/ReadVariableOpReadVariableOp#decoder_bn3_readvariableop_resource*
_output_shapes
:@*
dtype0~
decoder/bn3/ReadVariableOp_1ReadVariableOp%decoder_bn3_readvariableop_1_resource*
_output_shapes
:@*
dtype0�
+decoder/bn3/FusedBatchNormV3/ReadVariableOpReadVariableOp4decoder_bn3_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype0�
-decoder/bn3/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp6decoder_bn3_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype0�
decoder/bn3/FusedBatchNormV3FusedBatchNormV3)decoder/convTranspose2D3/BiasAdd:output:0"decoder/bn3/ReadVariableOp:value:0$decoder/bn3/ReadVariableOp_1:value:03decoder/bn3/FusedBatchNormV3/ReadVariableOp:value:05decoder/bn3/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:���������@:@:@:@:@:*
epsilon%o�:*
is_training( �
decoder/lReLU3/LeakyRelu	LeakyRelu decoder/bn3/FusedBatchNormV3:y:0*/
_output_shapes
:���������@*
alpha%���>�
decoder/convTranspose2D4/ShapeShape&decoder/lReLU3/LeakyRelu:activations:0*
T0*
_output_shapes
::��v
,decoder/convTranspose2D4/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: x
.decoder/convTranspose2D4/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:x
.decoder/convTranspose2D4/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
&decoder/convTranspose2D4/strided_sliceStridedSlice'decoder/convTranspose2D4/Shape:output:05decoder/convTranspose2D4/strided_slice/stack:output:07decoder/convTranspose2D4/strided_slice/stack_1:output:07decoder/convTranspose2D4/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskb
 decoder/convTranspose2D4/stack/1Const*
_output_shapes
: *
dtype0*
value	B :b
 decoder/convTranspose2D4/stack/2Const*
_output_shapes
: *
dtype0*
value	B :b
 decoder/convTranspose2D4/stack/3Const*
_output_shapes
: *
dtype0*
value	B :@�
decoder/convTranspose2D4/stackPack/decoder/convTranspose2D4/strided_slice:output:0)decoder/convTranspose2D4/stack/1:output:0)decoder/convTranspose2D4/stack/2:output:0)decoder/convTranspose2D4/stack/3:output:0*
N*
T0*
_output_shapes
:x
.decoder/convTranspose2D4/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: z
0decoder/convTranspose2D4/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:z
0decoder/convTranspose2D4/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
(decoder/convTranspose2D4/strided_slice_1StridedSlice'decoder/convTranspose2D4/stack:output:07decoder/convTranspose2D4/strided_slice_1/stack:output:09decoder/convTranspose2D4/strided_slice_1/stack_1:output:09decoder/convTranspose2D4/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask�
8decoder/convTranspose2D4/conv2d_transpose/ReadVariableOpReadVariableOpAdecoder_convtranspose2d4_conv2d_transpose_readvariableop_resource*&
_output_shapes
:@@*
dtype0�
)decoder/convTranspose2D4/conv2d_transposeConv2DBackpropInput'decoder/convTranspose2D4/stack:output:0@decoder/convTranspose2D4/conv2d_transpose/ReadVariableOp:value:0&decoder/lReLU3/LeakyRelu:activations:0*
T0*/
_output_shapes
:���������@*
paddingSAME*
strides
�
/decoder/convTranspose2D4/BiasAdd/ReadVariableOpReadVariableOp8decoder_convtranspose2d4_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0�
 decoder/convTranspose2D4/BiasAddBiasAdd2decoder/convTranspose2D4/conv2d_transpose:output:07decoder/convTranspose2D4/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@z
decoder/bn4/ReadVariableOpReadVariableOp#decoder_bn4_readvariableop_resource*
_output_shapes
:@*
dtype0~
decoder/bn4/ReadVariableOp_1ReadVariableOp%decoder_bn4_readvariableop_1_resource*
_output_shapes
:@*
dtype0�
+decoder/bn4/FusedBatchNormV3/ReadVariableOpReadVariableOp4decoder_bn4_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype0�
-decoder/bn4/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp6decoder_bn4_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype0�
decoder/bn4/FusedBatchNormV3FusedBatchNormV3)decoder/convTranspose2D4/BiasAdd:output:0"decoder/bn4/ReadVariableOp:value:0$decoder/bn4/ReadVariableOp_1:value:03decoder/bn4/FusedBatchNormV3/ReadVariableOp:value:05decoder/bn4/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:���������@:@:@:@:@:*
epsilon%o�:*
is_training( �
decoder/lReLU4/LeakyRelu	LeakyRelu decoder/bn4/FusedBatchNormV3:y:0*/
_output_shapes
:���������@*
alpha%���>�
decoder/convTranspose2D5/ShapeShape&decoder/lReLU4/LeakyRelu:activations:0*
T0*
_output_shapes
::��v
,decoder/convTranspose2D5/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: x
.decoder/convTranspose2D5/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:x
.decoder/convTranspose2D5/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
&decoder/convTranspose2D5/strided_sliceStridedSlice'decoder/convTranspose2D5/Shape:output:05decoder/convTranspose2D5/strided_slice/stack:output:07decoder/convTranspose2D5/strided_slice/stack_1:output:07decoder/convTranspose2D5/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskb
 decoder/convTranspose2D5/stack/1Const*
_output_shapes
: *
dtype0*
value	B :b
 decoder/convTranspose2D5/stack/2Const*
_output_shapes
: *
dtype0*
value	B :b
 decoder/convTranspose2D5/stack/3Const*
_output_shapes
: *
dtype0*
value	B : �
decoder/convTranspose2D5/stackPack/decoder/convTranspose2D5/strided_slice:output:0)decoder/convTranspose2D5/stack/1:output:0)decoder/convTranspose2D5/stack/2:output:0)decoder/convTranspose2D5/stack/3:output:0*
N*
T0*
_output_shapes
:x
.decoder/convTranspose2D5/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: z
0decoder/convTranspose2D5/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:z
0decoder/convTranspose2D5/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
(decoder/convTranspose2D5/strided_slice_1StridedSlice'decoder/convTranspose2D5/stack:output:07decoder/convTranspose2D5/strided_slice_1/stack:output:09decoder/convTranspose2D5/strided_slice_1/stack_1:output:09decoder/convTranspose2D5/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask�
8decoder/convTranspose2D5/conv2d_transpose/ReadVariableOpReadVariableOpAdecoder_convtranspose2d5_conv2d_transpose_readvariableop_resource*&
_output_shapes
: @*
dtype0�
)decoder/convTranspose2D5/conv2d_transposeConv2DBackpropInput'decoder/convTranspose2D5/stack:output:0@decoder/convTranspose2D5/conv2d_transpose/ReadVariableOp:value:0&decoder/lReLU4/LeakyRelu:activations:0*
T0*/
_output_shapes
:��������� *
paddingSAME*
strides
�
/decoder/convTranspose2D5/BiasAdd/ReadVariableOpReadVariableOp8decoder_convtranspose2d5_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0�
 decoder/convTranspose2D5/BiasAddBiasAdd2decoder/convTranspose2D5/conv2d_transpose:output:07decoder/convTranspose2D5/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:��������� z
decoder/bn5/ReadVariableOpReadVariableOp#decoder_bn5_readvariableop_resource*
_output_shapes
: *
dtype0~
decoder/bn5/ReadVariableOp_1ReadVariableOp%decoder_bn5_readvariableop_1_resource*
_output_shapes
: *
dtype0�
+decoder/bn5/FusedBatchNormV3/ReadVariableOpReadVariableOp4decoder_bn5_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype0�
-decoder/bn5/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp6decoder_bn5_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype0�
decoder/bn5/FusedBatchNormV3FusedBatchNormV3)decoder/convTranspose2D5/BiasAdd:output:0"decoder/bn5/ReadVariableOp:value:0$decoder/bn5/ReadVariableOp_1:value:03decoder/bn5/FusedBatchNormV3/ReadVariableOp:value:05decoder/bn5/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:��������� : : : : :*
epsilon%o�:*
is_training( �
decoder/lReLU5/LeakyRelu	LeakyRelu decoder/bn5/FusedBatchNormV3:y:0*/
_output_shapes
:��������� *
alpha%���>�
decoder/convTranspose2D6/ShapeShape&decoder/lReLU5/LeakyRelu:activations:0*
T0*
_output_shapes
::��v
,decoder/convTranspose2D6/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: x
.decoder/convTranspose2D6/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:x
.decoder/convTranspose2D6/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
&decoder/convTranspose2D6/strided_sliceStridedSlice'decoder/convTranspose2D6/Shape:output:05decoder/convTranspose2D6/strided_slice/stack:output:07decoder/convTranspose2D6/strided_slice/stack_1:output:07decoder/convTranspose2D6/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskb
 decoder/convTranspose2D6/stack/1Const*
_output_shapes
: *
dtype0*
value	B :b
 decoder/convTranspose2D6/stack/2Const*
_output_shapes
: *
dtype0*
value	B :b
 decoder/convTranspose2D6/stack/3Const*
_output_shapes
: *
dtype0*
value	B :�
decoder/convTranspose2D6/stackPack/decoder/convTranspose2D6/strided_slice:output:0)decoder/convTranspose2D6/stack/1:output:0)decoder/convTranspose2D6/stack/2:output:0)decoder/convTranspose2D6/stack/3:output:0*
N*
T0*
_output_shapes
:x
.decoder/convTranspose2D6/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: z
0decoder/convTranspose2D6/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:z
0decoder/convTranspose2D6/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
(decoder/convTranspose2D6/strided_slice_1StridedSlice'decoder/convTranspose2D6/stack:output:07decoder/convTranspose2D6/strided_slice_1/stack:output:09decoder/convTranspose2D6/strided_slice_1/stack_1:output:09decoder/convTranspose2D6/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask�
8decoder/convTranspose2D6/conv2d_transpose/ReadVariableOpReadVariableOpAdecoder_convtranspose2d6_conv2d_transpose_readvariableop_resource*&
_output_shapes
: *
dtype0�
)decoder/convTranspose2D6/conv2d_transposeConv2DBackpropInput'decoder/convTranspose2D6/stack:output:0@decoder/convTranspose2D6/conv2d_transpose/ReadVariableOp:value:0&decoder/lReLU5/LeakyRelu:activations:0*
T0*/
_output_shapes
:���������*
paddingSAME*
strides
�
/decoder/convTranspose2D6/BiasAdd/ReadVariableOpReadVariableOp8decoder_convtranspose2d6_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
 decoder/convTranspose2D6/BiasAddBiasAdd2decoder/convTranspose2D6/conv2d_transpose:output:07decoder/convTranspose2D6/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:����������
 decoder/convTranspose2D6/SigmoidSigmoid)decoder/convTranspose2D6/BiasAdd:output:0*
T0*/
_output_shapes
:���������{
IdentityIdentity$decoder/convTranspose2D6/Sigmoid:y:0^NoOp*
T0*/
_output_shapes
:����������
NoOpNoOp,^decoder/bn1/FusedBatchNormV3/ReadVariableOp.^decoder/bn1/FusedBatchNormV3/ReadVariableOp_1^decoder/bn1/ReadVariableOp^decoder/bn1/ReadVariableOp_1,^decoder/bn2/FusedBatchNormV3/ReadVariableOp.^decoder/bn2/FusedBatchNormV3/ReadVariableOp_1^decoder/bn2/ReadVariableOp^decoder/bn2/ReadVariableOp_1,^decoder/bn3/FusedBatchNormV3/ReadVariableOp.^decoder/bn3/FusedBatchNormV3/ReadVariableOp_1^decoder/bn3/ReadVariableOp^decoder/bn3/ReadVariableOp_1,^decoder/bn4/FusedBatchNormV3/ReadVariableOp.^decoder/bn4/FusedBatchNormV3/ReadVariableOp_1^decoder/bn4/ReadVariableOp^decoder/bn4/ReadVariableOp_1,^decoder/bn5/FusedBatchNormV3/ReadVariableOp.^decoder/bn5/FusedBatchNormV3/ReadVariableOp_1^decoder/bn5/ReadVariableOp^decoder/bn5/ReadVariableOp_10^decoder/convTranspose2D1/BiasAdd/ReadVariableOp9^decoder/convTranspose2D1/conv2d_transpose/ReadVariableOp0^decoder/convTranspose2D2/BiasAdd/ReadVariableOp9^decoder/convTranspose2D2/conv2d_transpose/ReadVariableOp0^decoder/convTranspose2D3/BiasAdd/ReadVariableOp9^decoder/convTranspose2D3/conv2d_transpose/ReadVariableOp0^decoder/convTranspose2D4/BiasAdd/ReadVariableOp9^decoder/convTranspose2D4/conv2d_transpose/ReadVariableOp0^decoder/convTranspose2D5/BiasAdd/ReadVariableOp9^decoder/convTranspose2D5/conv2d_transpose/ReadVariableOp0^decoder/convTranspose2D6/BiasAdd/ReadVariableOp9^decoder/convTranspose2D6/conv2d_transpose/ReadVariableOp&^decoder/dense1/BiasAdd/ReadVariableOp%^decoder/dense1/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*j
_input_shapesY
W:���������
: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2Z
+decoder/bn1/FusedBatchNormV3/ReadVariableOp+decoder/bn1/FusedBatchNormV3/ReadVariableOp2^
-decoder/bn1/FusedBatchNormV3/ReadVariableOp_1-decoder/bn1/FusedBatchNormV3/ReadVariableOp_128
decoder/bn1/ReadVariableOpdecoder/bn1/ReadVariableOp2<
decoder/bn1/ReadVariableOp_1decoder/bn1/ReadVariableOp_12Z
+decoder/bn2/FusedBatchNormV3/ReadVariableOp+decoder/bn2/FusedBatchNormV3/ReadVariableOp2^
-decoder/bn2/FusedBatchNormV3/ReadVariableOp_1-decoder/bn2/FusedBatchNormV3/ReadVariableOp_128
decoder/bn2/ReadVariableOpdecoder/bn2/ReadVariableOp2<
decoder/bn2/ReadVariableOp_1decoder/bn2/ReadVariableOp_12Z
+decoder/bn3/FusedBatchNormV3/ReadVariableOp+decoder/bn3/FusedBatchNormV3/ReadVariableOp2^
-decoder/bn3/FusedBatchNormV3/ReadVariableOp_1-decoder/bn3/FusedBatchNormV3/ReadVariableOp_128
decoder/bn3/ReadVariableOpdecoder/bn3/ReadVariableOp2<
decoder/bn3/ReadVariableOp_1decoder/bn3/ReadVariableOp_12Z
+decoder/bn4/FusedBatchNormV3/ReadVariableOp+decoder/bn4/FusedBatchNormV3/ReadVariableOp2^
-decoder/bn4/FusedBatchNormV3/ReadVariableOp_1-decoder/bn4/FusedBatchNormV3/ReadVariableOp_128
decoder/bn4/ReadVariableOpdecoder/bn4/ReadVariableOp2<
decoder/bn4/ReadVariableOp_1decoder/bn4/ReadVariableOp_12Z
+decoder/bn5/FusedBatchNormV3/ReadVariableOp+decoder/bn5/FusedBatchNormV3/ReadVariableOp2^
-decoder/bn5/FusedBatchNormV3/ReadVariableOp_1-decoder/bn5/FusedBatchNormV3/ReadVariableOp_128
decoder/bn5/ReadVariableOpdecoder/bn5/ReadVariableOp2<
decoder/bn5/ReadVariableOp_1decoder/bn5/ReadVariableOp_12b
/decoder/convTranspose2D1/BiasAdd/ReadVariableOp/decoder/convTranspose2D1/BiasAdd/ReadVariableOp2t
8decoder/convTranspose2D1/conv2d_transpose/ReadVariableOp8decoder/convTranspose2D1/conv2d_transpose/ReadVariableOp2b
/decoder/convTranspose2D2/BiasAdd/ReadVariableOp/decoder/convTranspose2D2/BiasAdd/ReadVariableOp2t
8decoder/convTranspose2D2/conv2d_transpose/ReadVariableOp8decoder/convTranspose2D2/conv2d_transpose/ReadVariableOp2b
/decoder/convTranspose2D3/BiasAdd/ReadVariableOp/decoder/convTranspose2D3/BiasAdd/ReadVariableOp2t
8decoder/convTranspose2D3/conv2d_transpose/ReadVariableOp8decoder/convTranspose2D3/conv2d_transpose/ReadVariableOp2b
/decoder/convTranspose2D4/BiasAdd/ReadVariableOp/decoder/convTranspose2D4/BiasAdd/ReadVariableOp2t
8decoder/convTranspose2D4/conv2d_transpose/ReadVariableOp8decoder/convTranspose2D4/conv2d_transpose/ReadVariableOp2b
/decoder/convTranspose2D5/BiasAdd/ReadVariableOp/decoder/convTranspose2D5/BiasAdd/ReadVariableOp2t
8decoder/convTranspose2D5/conv2d_transpose/ReadVariableOp8decoder/convTranspose2D5/conv2d_transpose/ReadVariableOp2b
/decoder/convTranspose2D6/BiasAdd/ReadVariableOp/decoder/convTranspose2D6/BiasAdd/ReadVariableOp2t
8decoder/convTranspose2D6/conv2d_transpose/ReadVariableOp8decoder/convTranspose2D6/conv2d_transpose/ReadVariableOp2N
%decoder/dense1/BiasAdd/ReadVariableOp%decoder/dense1/BiasAdd/ReadVariableOp2L
$decoder/dense1/MatMul/ReadVariableOp$decoder/dense1/MatMul/ReadVariableOp:S O
'
_output_shapes
:���������

$
_user_specified_name
inputLayer
��
�
B__inference_decoder_layer_call_and_return_conditional_losses_16844

inputs8
%dense1_matmul_readvariableop_resource:	
�5
&dense1_biasadd_readvariableop_resource:	�S
9convtranspose2d1_conv2d_transpose_readvariableop_resource:  >
0convtranspose2d1_biasadd_readvariableop_resource: )
bn1_readvariableop_resource: +
bn1_readvariableop_1_resource: :
,bn1_fusedbatchnormv3_readvariableop_resource: <
.bn1_fusedbatchnormv3_readvariableop_1_resource: S
9convtranspose2d2_conv2d_transpose_readvariableop_resource:@ >
0convtranspose2d2_biasadd_readvariableop_resource:@)
bn2_readvariableop_resource:@+
bn2_readvariableop_1_resource:@:
,bn2_fusedbatchnormv3_readvariableop_resource:@<
.bn2_fusedbatchnormv3_readvariableop_1_resource:@S
9convtranspose2d3_conv2d_transpose_readvariableop_resource:@@>
0convtranspose2d3_biasadd_readvariableop_resource:@)
bn3_readvariableop_resource:@+
bn3_readvariableop_1_resource:@:
,bn3_fusedbatchnormv3_readvariableop_resource:@<
.bn3_fusedbatchnormv3_readvariableop_1_resource:@S
9convtranspose2d4_conv2d_transpose_readvariableop_resource:@@>
0convtranspose2d4_biasadd_readvariableop_resource:@)
bn4_readvariableop_resource:@+
bn4_readvariableop_1_resource:@:
,bn4_fusedbatchnormv3_readvariableop_resource:@<
.bn4_fusedbatchnormv3_readvariableop_1_resource:@S
9convtranspose2d5_conv2d_transpose_readvariableop_resource: @>
0convtranspose2d5_biasadd_readvariableop_resource: )
bn5_readvariableop_resource: +
bn5_readvariableop_1_resource: :
,bn5_fusedbatchnormv3_readvariableop_resource: <
.bn5_fusedbatchnormv3_readvariableop_1_resource: S
9convtranspose2d6_conv2d_transpose_readvariableop_resource: >
0convtranspose2d6_biasadd_readvariableop_resource:
identity��#bn1/FusedBatchNormV3/ReadVariableOp�%bn1/FusedBatchNormV3/ReadVariableOp_1�bn1/ReadVariableOp�bn1/ReadVariableOp_1�#bn2/FusedBatchNormV3/ReadVariableOp�%bn2/FusedBatchNormV3/ReadVariableOp_1�bn2/ReadVariableOp�bn2/ReadVariableOp_1�#bn3/FusedBatchNormV3/ReadVariableOp�%bn3/FusedBatchNormV3/ReadVariableOp_1�bn3/ReadVariableOp�bn3/ReadVariableOp_1�#bn4/FusedBatchNormV3/ReadVariableOp�%bn4/FusedBatchNormV3/ReadVariableOp_1�bn4/ReadVariableOp�bn4/ReadVariableOp_1�#bn5/FusedBatchNormV3/ReadVariableOp�%bn5/FusedBatchNormV3/ReadVariableOp_1�bn5/ReadVariableOp�bn5/ReadVariableOp_1�'convTranspose2D1/BiasAdd/ReadVariableOp�0convTranspose2D1/conv2d_transpose/ReadVariableOp�'convTranspose2D2/BiasAdd/ReadVariableOp�0convTranspose2D2/conv2d_transpose/ReadVariableOp�'convTranspose2D3/BiasAdd/ReadVariableOp�0convTranspose2D3/conv2d_transpose/ReadVariableOp�'convTranspose2D4/BiasAdd/ReadVariableOp�0convTranspose2D4/conv2d_transpose/ReadVariableOp�'convTranspose2D5/BiasAdd/ReadVariableOp�0convTranspose2D5/conv2d_transpose/ReadVariableOp�'convTranspose2D6/BiasAdd/ReadVariableOp�0convTranspose2D6/conv2d_transpose/ReadVariableOp�dense1/BiasAdd/ReadVariableOp�dense1/MatMul/ReadVariableOp�
dense1/MatMul/ReadVariableOpReadVariableOp%dense1_matmul_readvariableop_resource*
_output_shapes
:	
�*
dtype0x
dense1/MatMulMatMulinputs$dense1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
dense1/BiasAdd/ReadVariableOpReadVariableOp&dense1_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
dense1/BiasAddBiasAdddense1/MatMul:product:0%dense1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������g
reshapeLayer/ShapeShapedense1/BiasAdd:output:0*
T0*
_output_shapes
::��j
 reshapeLayer/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: l
"reshapeLayer/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:l
"reshapeLayer/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
reshapeLayer/strided_sliceStridedSlicereshapeLayer/Shape:output:0)reshapeLayer/strided_slice/stack:output:0+reshapeLayer/strided_slice/stack_1:output:0+reshapeLayer/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask^
reshapeLayer/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :^
reshapeLayer/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :^
reshapeLayer/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B : �
reshapeLayer/Reshape/shapePack#reshapeLayer/strided_slice:output:0%reshapeLayer/Reshape/shape/1:output:0%reshapeLayer/Reshape/shape/2:output:0%reshapeLayer/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:�
reshapeLayer/ReshapeReshapedense1/BiasAdd:output:0#reshapeLayer/Reshape/shape:output:0*
T0*/
_output_shapes
:��������� q
convTranspose2D1/ShapeShapereshapeLayer/Reshape:output:0*
T0*
_output_shapes
::��n
$convTranspose2D1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: p
&convTranspose2D1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:p
&convTranspose2D1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
convTranspose2D1/strided_sliceStridedSliceconvTranspose2D1/Shape:output:0-convTranspose2D1/strided_slice/stack:output:0/convTranspose2D1/strided_slice/stack_1:output:0/convTranspose2D1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskZ
convTranspose2D1/stack/1Const*
_output_shapes
: *
dtype0*
value	B :Z
convTranspose2D1/stack/2Const*
_output_shapes
: *
dtype0*
value	B :Z
convTranspose2D1/stack/3Const*
_output_shapes
: *
dtype0*
value	B : �
convTranspose2D1/stackPack'convTranspose2D1/strided_slice:output:0!convTranspose2D1/stack/1:output:0!convTranspose2D1/stack/2:output:0!convTranspose2D1/stack/3:output:0*
N*
T0*
_output_shapes
:p
&convTranspose2D1/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: r
(convTranspose2D1/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:r
(convTranspose2D1/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
 convTranspose2D1/strided_slice_1StridedSliceconvTranspose2D1/stack:output:0/convTranspose2D1/strided_slice_1/stack:output:01convTranspose2D1/strided_slice_1/stack_1:output:01convTranspose2D1/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask�
0convTranspose2D1/conv2d_transpose/ReadVariableOpReadVariableOp9convtranspose2d1_conv2d_transpose_readvariableop_resource*&
_output_shapes
:  *
dtype0�
!convTranspose2D1/conv2d_transposeConv2DBackpropInputconvTranspose2D1/stack:output:08convTranspose2D1/conv2d_transpose/ReadVariableOp:value:0reshapeLayer/Reshape:output:0*
T0*/
_output_shapes
:��������� *
paddingSAME*
strides
�
'convTranspose2D1/BiasAdd/ReadVariableOpReadVariableOp0convtranspose2d1_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0�
convTranspose2D1/BiasAddBiasAdd*convTranspose2D1/conv2d_transpose:output:0/convTranspose2D1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:��������� j
bn1/ReadVariableOpReadVariableOpbn1_readvariableop_resource*
_output_shapes
: *
dtype0n
bn1/ReadVariableOp_1ReadVariableOpbn1_readvariableop_1_resource*
_output_shapes
: *
dtype0�
#bn1/FusedBatchNormV3/ReadVariableOpReadVariableOp,bn1_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype0�
%bn1/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp.bn1_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype0�
bn1/FusedBatchNormV3FusedBatchNormV3!convTranspose2D1/BiasAdd:output:0bn1/ReadVariableOp:value:0bn1/ReadVariableOp_1:value:0+bn1/FusedBatchNormV3/ReadVariableOp:value:0-bn1/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:��������� : : : : :*
epsilon%o�:*
is_training( x
lReLU1/LeakyRelu	LeakyRelubn1/FusedBatchNormV3:y:0*/
_output_shapes
:��������� *
alpha%���>r
convTranspose2D2/ShapeShapelReLU1/LeakyRelu:activations:0*
T0*
_output_shapes
::��n
$convTranspose2D2/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: p
&convTranspose2D2/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:p
&convTranspose2D2/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
convTranspose2D2/strided_sliceStridedSliceconvTranspose2D2/Shape:output:0-convTranspose2D2/strided_slice/stack:output:0/convTranspose2D2/strided_slice/stack_1:output:0/convTranspose2D2/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskZ
convTranspose2D2/stack/1Const*
_output_shapes
: *
dtype0*
value	B :Z
convTranspose2D2/stack/2Const*
_output_shapes
: *
dtype0*
value	B :Z
convTranspose2D2/stack/3Const*
_output_shapes
: *
dtype0*
value	B :@�
convTranspose2D2/stackPack'convTranspose2D2/strided_slice:output:0!convTranspose2D2/stack/1:output:0!convTranspose2D2/stack/2:output:0!convTranspose2D2/stack/3:output:0*
N*
T0*
_output_shapes
:p
&convTranspose2D2/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: r
(convTranspose2D2/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:r
(convTranspose2D2/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
 convTranspose2D2/strided_slice_1StridedSliceconvTranspose2D2/stack:output:0/convTranspose2D2/strided_slice_1/stack:output:01convTranspose2D2/strided_slice_1/stack_1:output:01convTranspose2D2/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask�
0convTranspose2D2/conv2d_transpose/ReadVariableOpReadVariableOp9convtranspose2d2_conv2d_transpose_readvariableop_resource*&
_output_shapes
:@ *
dtype0�
!convTranspose2D2/conv2d_transposeConv2DBackpropInputconvTranspose2D2/stack:output:08convTranspose2D2/conv2d_transpose/ReadVariableOp:value:0lReLU1/LeakyRelu:activations:0*
T0*/
_output_shapes
:���������@*
paddingSAME*
strides
�
'convTranspose2D2/BiasAdd/ReadVariableOpReadVariableOp0convtranspose2d2_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0�
convTranspose2D2/BiasAddBiasAdd*convTranspose2D2/conv2d_transpose:output:0/convTranspose2D2/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@j
bn2/ReadVariableOpReadVariableOpbn2_readvariableop_resource*
_output_shapes
:@*
dtype0n
bn2/ReadVariableOp_1ReadVariableOpbn2_readvariableop_1_resource*
_output_shapes
:@*
dtype0�
#bn2/FusedBatchNormV3/ReadVariableOpReadVariableOp,bn2_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype0�
%bn2/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp.bn2_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype0�
bn2/FusedBatchNormV3FusedBatchNormV3!convTranspose2D2/BiasAdd:output:0bn2/ReadVariableOp:value:0bn2/ReadVariableOp_1:value:0+bn2/FusedBatchNormV3/ReadVariableOp:value:0-bn2/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:���������@:@:@:@:@:*
epsilon%o�:*
is_training( x
lReLU2/LeakyRelu	LeakyRelubn2/FusedBatchNormV3:y:0*/
_output_shapes
:���������@*
alpha%���>r
convTranspose2D3/ShapeShapelReLU2/LeakyRelu:activations:0*
T0*
_output_shapes
::��n
$convTranspose2D3/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: p
&convTranspose2D3/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:p
&convTranspose2D3/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
convTranspose2D3/strided_sliceStridedSliceconvTranspose2D3/Shape:output:0-convTranspose2D3/strided_slice/stack:output:0/convTranspose2D3/strided_slice/stack_1:output:0/convTranspose2D3/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskZ
convTranspose2D3/stack/1Const*
_output_shapes
: *
dtype0*
value	B :Z
convTranspose2D3/stack/2Const*
_output_shapes
: *
dtype0*
value	B :Z
convTranspose2D3/stack/3Const*
_output_shapes
: *
dtype0*
value	B :@�
convTranspose2D3/stackPack'convTranspose2D3/strided_slice:output:0!convTranspose2D3/stack/1:output:0!convTranspose2D3/stack/2:output:0!convTranspose2D3/stack/3:output:0*
N*
T0*
_output_shapes
:p
&convTranspose2D3/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: r
(convTranspose2D3/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:r
(convTranspose2D3/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
 convTranspose2D3/strided_slice_1StridedSliceconvTranspose2D3/stack:output:0/convTranspose2D3/strided_slice_1/stack:output:01convTranspose2D3/strided_slice_1/stack_1:output:01convTranspose2D3/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask�
0convTranspose2D3/conv2d_transpose/ReadVariableOpReadVariableOp9convtranspose2d3_conv2d_transpose_readvariableop_resource*&
_output_shapes
:@@*
dtype0�
!convTranspose2D3/conv2d_transposeConv2DBackpropInputconvTranspose2D3/stack:output:08convTranspose2D3/conv2d_transpose/ReadVariableOp:value:0lReLU2/LeakyRelu:activations:0*
T0*/
_output_shapes
:���������@*
paddingSAME*
strides
�
'convTranspose2D3/BiasAdd/ReadVariableOpReadVariableOp0convtranspose2d3_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0�
convTranspose2D3/BiasAddBiasAdd*convTranspose2D3/conv2d_transpose:output:0/convTranspose2D3/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@j
bn3/ReadVariableOpReadVariableOpbn3_readvariableop_resource*
_output_shapes
:@*
dtype0n
bn3/ReadVariableOp_1ReadVariableOpbn3_readvariableop_1_resource*
_output_shapes
:@*
dtype0�
#bn3/FusedBatchNormV3/ReadVariableOpReadVariableOp,bn3_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype0�
%bn3/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp.bn3_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype0�
bn3/FusedBatchNormV3FusedBatchNormV3!convTranspose2D3/BiasAdd:output:0bn3/ReadVariableOp:value:0bn3/ReadVariableOp_1:value:0+bn3/FusedBatchNormV3/ReadVariableOp:value:0-bn3/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:���������@:@:@:@:@:*
epsilon%o�:*
is_training( x
lReLU3/LeakyRelu	LeakyRelubn3/FusedBatchNormV3:y:0*/
_output_shapes
:���������@*
alpha%���>r
convTranspose2D4/ShapeShapelReLU3/LeakyRelu:activations:0*
T0*
_output_shapes
::��n
$convTranspose2D4/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: p
&convTranspose2D4/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:p
&convTranspose2D4/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
convTranspose2D4/strided_sliceStridedSliceconvTranspose2D4/Shape:output:0-convTranspose2D4/strided_slice/stack:output:0/convTranspose2D4/strided_slice/stack_1:output:0/convTranspose2D4/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskZ
convTranspose2D4/stack/1Const*
_output_shapes
: *
dtype0*
value	B :Z
convTranspose2D4/stack/2Const*
_output_shapes
: *
dtype0*
value	B :Z
convTranspose2D4/stack/3Const*
_output_shapes
: *
dtype0*
value	B :@�
convTranspose2D4/stackPack'convTranspose2D4/strided_slice:output:0!convTranspose2D4/stack/1:output:0!convTranspose2D4/stack/2:output:0!convTranspose2D4/stack/3:output:0*
N*
T0*
_output_shapes
:p
&convTranspose2D4/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: r
(convTranspose2D4/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:r
(convTranspose2D4/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
 convTranspose2D4/strided_slice_1StridedSliceconvTranspose2D4/stack:output:0/convTranspose2D4/strided_slice_1/stack:output:01convTranspose2D4/strided_slice_1/stack_1:output:01convTranspose2D4/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask�
0convTranspose2D4/conv2d_transpose/ReadVariableOpReadVariableOp9convtranspose2d4_conv2d_transpose_readvariableop_resource*&
_output_shapes
:@@*
dtype0�
!convTranspose2D4/conv2d_transposeConv2DBackpropInputconvTranspose2D4/stack:output:08convTranspose2D4/conv2d_transpose/ReadVariableOp:value:0lReLU3/LeakyRelu:activations:0*
T0*/
_output_shapes
:���������@*
paddingSAME*
strides
�
'convTranspose2D4/BiasAdd/ReadVariableOpReadVariableOp0convtranspose2d4_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0�
convTranspose2D4/BiasAddBiasAdd*convTranspose2D4/conv2d_transpose:output:0/convTranspose2D4/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@j
bn4/ReadVariableOpReadVariableOpbn4_readvariableop_resource*
_output_shapes
:@*
dtype0n
bn4/ReadVariableOp_1ReadVariableOpbn4_readvariableop_1_resource*
_output_shapes
:@*
dtype0�
#bn4/FusedBatchNormV3/ReadVariableOpReadVariableOp,bn4_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype0�
%bn4/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp.bn4_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype0�
bn4/FusedBatchNormV3FusedBatchNormV3!convTranspose2D4/BiasAdd:output:0bn4/ReadVariableOp:value:0bn4/ReadVariableOp_1:value:0+bn4/FusedBatchNormV3/ReadVariableOp:value:0-bn4/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:���������@:@:@:@:@:*
epsilon%o�:*
is_training( x
lReLU4/LeakyRelu	LeakyRelubn4/FusedBatchNormV3:y:0*/
_output_shapes
:���������@*
alpha%���>r
convTranspose2D5/ShapeShapelReLU4/LeakyRelu:activations:0*
T0*
_output_shapes
::��n
$convTranspose2D5/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: p
&convTranspose2D5/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:p
&convTranspose2D5/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
convTranspose2D5/strided_sliceStridedSliceconvTranspose2D5/Shape:output:0-convTranspose2D5/strided_slice/stack:output:0/convTranspose2D5/strided_slice/stack_1:output:0/convTranspose2D5/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskZ
convTranspose2D5/stack/1Const*
_output_shapes
: *
dtype0*
value	B :Z
convTranspose2D5/stack/2Const*
_output_shapes
: *
dtype0*
value	B :Z
convTranspose2D5/stack/3Const*
_output_shapes
: *
dtype0*
value	B : �
convTranspose2D5/stackPack'convTranspose2D5/strided_slice:output:0!convTranspose2D5/stack/1:output:0!convTranspose2D5/stack/2:output:0!convTranspose2D5/stack/3:output:0*
N*
T0*
_output_shapes
:p
&convTranspose2D5/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: r
(convTranspose2D5/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:r
(convTranspose2D5/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
 convTranspose2D5/strided_slice_1StridedSliceconvTranspose2D5/stack:output:0/convTranspose2D5/strided_slice_1/stack:output:01convTranspose2D5/strided_slice_1/stack_1:output:01convTranspose2D5/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask�
0convTranspose2D5/conv2d_transpose/ReadVariableOpReadVariableOp9convtranspose2d5_conv2d_transpose_readvariableop_resource*&
_output_shapes
: @*
dtype0�
!convTranspose2D5/conv2d_transposeConv2DBackpropInputconvTranspose2D5/stack:output:08convTranspose2D5/conv2d_transpose/ReadVariableOp:value:0lReLU4/LeakyRelu:activations:0*
T0*/
_output_shapes
:��������� *
paddingSAME*
strides
�
'convTranspose2D5/BiasAdd/ReadVariableOpReadVariableOp0convtranspose2d5_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0�
convTranspose2D5/BiasAddBiasAdd*convTranspose2D5/conv2d_transpose:output:0/convTranspose2D5/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:��������� j
bn5/ReadVariableOpReadVariableOpbn5_readvariableop_resource*
_output_shapes
: *
dtype0n
bn5/ReadVariableOp_1ReadVariableOpbn5_readvariableop_1_resource*
_output_shapes
: *
dtype0�
#bn5/FusedBatchNormV3/ReadVariableOpReadVariableOp,bn5_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype0�
%bn5/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp.bn5_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype0�
bn5/FusedBatchNormV3FusedBatchNormV3!convTranspose2D5/BiasAdd:output:0bn5/ReadVariableOp:value:0bn5/ReadVariableOp_1:value:0+bn5/FusedBatchNormV3/ReadVariableOp:value:0-bn5/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:��������� : : : : :*
epsilon%o�:*
is_training( x
lReLU5/LeakyRelu	LeakyRelubn5/FusedBatchNormV3:y:0*/
_output_shapes
:��������� *
alpha%���>r
convTranspose2D6/ShapeShapelReLU5/LeakyRelu:activations:0*
T0*
_output_shapes
::��n
$convTranspose2D6/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: p
&convTranspose2D6/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:p
&convTranspose2D6/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
convTranspose2D6/strided_sliceStridedSliceconvTranspose2D6/Shape:output:0-convTranspose2D6/strided_slice/stack:output:0/convTranspose2D6/strided_slice/stack_1:output:0/convTranspose2D6/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskZ
convTranspose2D6/stack/1Const*
_output_shapes
: *
dtype0*
value	B :Z
convTranspose2D6/stack/2Const*
_output_shapes
: *
dtype0*
value	B :Z
convTranspose2D6/stack/3Const*
_output_shapes
: *
dtype0*
value	B :�
convTranspose2D6/stackPack'convTranspose2D6/strided_slice:output:0!convTranspose2D6/stack/1:output:0!convTranspose2D6/stack/2:output:0!convTranspose2D6/stack/3:output:0*
N*
T0*
_output_shapes
:p
&convTranspose2D6/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: r
(convTranspose2D6/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:r
(convTranspose2D6/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
 convTranspose2D6/strided_slice_1StridedSliceconvTranspose2D6/stack:output:0/convTranspose2D6/strided_slice_1/stack:output:01convTranspose2D6/strided_slice_1/stack_1:output:01convTranspose2D6/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask�
0convTranspose2D6/conv2d_transpose/ReadVariableOpReadVariableOp9convtranspose2d6_conv2d_transpose_readvariableop_resource*&
_output_shapes
: *
dtype0�
!convTranspose2D6/conv2d_transposeConv2DBackpropInputconvTranspose2D6/stack:output:08convTranspose2D6/conv2d_transpose/ReadVariableOp:value:0lReLU5/LeakyRelu:activations:0*
T0*/
_output_shapes
:���������*
paddingSAME*
strides
�
'convTranspose2D6/BiasAdd/ReadVariableOpReadVariableOp0convtranspose2d6_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
convTranspose2D6/BiasAddBiasAdd*convTranspose2D6/conv2d_transpose:output:0/convTranspose2D6/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:����������
convTranspose2D6/SigmoidSigmoid!convTranspose2D6/BiasAdd:output:0*
T0*/
_output_shapes
:���������s
IdentityIdentityconvTranspose2D6/Sigmoid:y:0^NoOp*
T0*/
_output_shapes
:����������

NoOpNoOp$^bn1/FusedBatchNormV3/ReadVariableOp&^bn1/FusedBatchNormV3/ReadVariableOp_1^bn1/ReadVariableOp^bn1/ReadVariableOp_1$^bn2/FusedBatchNormV3/ReadVariableOp&^bn2/FusedBatchNormV3/ReadVariableOp_1^bn2/ReadVariableOp^bn2/ReadVariableOp_1$^bn3/FusedBatchNormV3/ReadVariableOp&^bn3/FusedBatchNormV3/ReadVariableOp_1^bn3/ReadVariableOp^bn3/ReadVariableOp_1$^bn4/FusedBatchNormV3/ReadVariableOp&^bn4/FusedBatchNormV3/ReadVariableOp_1^bn4/ReadVariableOp^bn4/ReadVariableOp_1$^bn5/FusedBatchNormV3/ReadVariableOp&^bn5/FusedBatchNormV3/ReadVariableOp_1^bn5/ReadVariableOp^bn5/ReadVariableOp_1(^convTranspose2D1/BiasAdd/ReadVariableOp1^convTranspose2D1/conv2d_transpose/ReadVariableOp(^convTranspose2D2/BiasAdd/ReadVariableOp1^convTranspose2D2/conv2d_transpose/ReadVariableOp(^convTranspose2D3/BiasAdd/ReadVariableOp1^convTranspose2D3/conv2d_transpose/ReadVariableOp(^convTranspose2D4/BiasAdd/ReadVariableOp1^convTranspose2D4/conv2d_transpose/ReadVariableOp(^convTranspose2D5/BiasAdd/ReadVariableOp1^convTranspose2D5/conv2d_transpose/ReadVariableOp(^convTranspose2D6/BiasAdd/ReadVariableOp1^convTranspose2D6/conv2d_transpose/ReadVariableOp^dense1/BiasAdd/ReadVariableOp^dense1/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*j
_input_shapesY
W:���������
: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2J
#bn1/FusedBatchNormV3/ReadVariableOp#bn1/FusedBatchNormV3/ReadVariableOp2N
%bn1/FusedBatchNormV3/ReadVariableOp_1%bn1/FusedBatchNormV3/ReadVariableOp_12(
bn1/ReadVariableOpbn1/ReadVariableOp2,
bn1/ReadVariableOp_1bn1/ReadVariableOp_12J
#bn2/FusedBatchNormV3/ReadVariableOp#bn2/FusedBatchNormV3/ReadVariableOp2N
%bn2/FusedBatchNormV3/ReadVariableOp_1%bn2/FusedBatchNormV3/ReadVariableOp_12(
bn2/ReadVariableOpbn2/ReadVariableOp2,
bn2/ReadVariableOp_1bn2/ReadVariableOp_12J
#bn3/FusedBatchNormV3/ReadVariableOp#bn3/FusedBatchNormV3/ReadVariableOp2N
%bn3/FusedBatchNormV3/ReadVariableOp_1%bn3/FusedBatchNormV3/ReadVariableOp_12(
bn3/ReadVariableOpbn3/ReadVariableOp2,
bn3/ReadVariableOp_1bn3/ReadVariableOp_12J
#bn4/FusedBatchNormV3/ReadVariableOp#bn4/FusedBatchNormV3/ReadVariableOp2N
%bn4/FusedBatchNormV3/ReadVariableOp_1%bn4/FusedBatchNormV3/ReadVariableOp_12(
bn4/ReadVariableOpbn4/ReadVariableOp2,
bn4/ReadVariableOp_1bn4/ReadVariableOp_12J
#bn5/FusedBatchNormV3/ReadVariableOp#bn5/FusedBatchNormV3/ReadVariableOp2N
%bn5/FusedBatchNormV3/ReadVariableOp_1%bn5/FusedBatchNormV3/ReadVariableOp_12(
bn5/ReadVariableOpbn5/ReadVariableOp2,
bn5/ReadVariableOp_1bn5/ReadVariableOp_12R
'convTranspose2D1/BiasAdd/ReadVariableOp'convTranspose2D1/BiasAdd/ReadVariableOp2d
0convTranspose2D1/conv2d_transpose/ReadVariableOp0convTranspose2D1/conv2d_transpose/ReadVariableOp2R
'convTranspose2D2/BiasAdd/ReadVariableOp'convTranspose2D2/BiasAdd/ReadVariableOp2d
0convTranspose2D2/conv2d_transpose/ReadVariableOp0convTranspose2D2/conv2d_transpose/ReadVariableOp2R
'convTranspose2D3/BiasAdd/ReadVariableOp'convTranspose2D3/BiasAdd/ReadVariableOp2d
0convTranspose2D3/conv2d_transpose/ReadVariableOp0convTranspose2D3/conv2d_transpose/ReadVariableOp2R
'convTranspose2D4/BiasAdd/ReadVariableOp'convTranspose2D4/BiasAdd/ReadVariableOp2d
0convTranspose2D4/conv2d_transpose/ReadVariableOp0convTranspose2D4/conv2d_transpose/ReadVariableOp2R
'convTranspose2D5/BiasAdd/ReadVariableOp'convTranspose2D5/BiasAdd/ReadVariableOp2d
0convTranspose2D5/conv2d_transpose/ReadVariableOp0convTranspose2D5/conv2d_transpose/ReadVariableOp2R
'convTranspose2D6/BiasAdd/ReadVariableOp'convTranspose2D6/BiasAdd/ReadVariableOp2d
0convTranspose2D6/conv2d_transpose/ReadVariableOp0convTranspose2D6/conv2d_transpose/ReadVariableOp2>
dense1/BiasAdd/ReadVariableOpdense1/BiasAdd/ReadVariableOp2<
dense1/MatMul/ReadVariableOpdense1/MatMul/ReadVariableOp:O K
'
_output_shapes
:���������

 
_user_specified_nameinputs
�
�
>__inference_bn1_layer_call_and_return_conditional_losses_14943

inputs%
readvariableop_resource: '
readvariableop_1_resource: 6
(fusedbatchnormv3_readvariableop_resource: 8
*fusedbatchnormv3_readvariableop_1_resource: 
identity��FusedBatchNormV3/ReadVariableOp�!FusedBatchNormV3/ReadVariableOp_1�ReadVariableOp�ReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype0�
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype0�
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype0�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+��������������������������� : : : : :*
epsilon%o�:*
is_training( }
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+��������������������������� �
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+��������������������������� : : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+��������������������������� 
 
_user_specified_nameinputs
��
�
B__inference_decoder_layer_call_and_return_conditional_losses_16634

inputs8
%dense1_matmul_readvariableop_resource:	
�5
&dense1_biasadd_readvariableop_resource:	�S
9convtranspose2d1_conv2d_transpose_readvariableop_resource:  >
0convtranspose2d1_biasadd_readvariableop_resource: )
bn1_readvariableop_resource: +
bn1_readvariableop_1_resource: :
,bn1_fusedbatchnormv3_readvariableop_resource: <
.bn1_fusedbatchnormv3_readvariableop_1_resource: S
9convtranspose2d2_conv2d_transpose_readvariableop_resource:@ >
0convtranspose2d2_biasadd_readvariableop_resource:@)
bn2_readvariableop_resource:@+
bn2_readvariableop_1_resource:@:
,bn2_fusedbatchnormv3_readvariableop_resource:@<
.bn2_fusedbatchnormv3_readvariableop_1_resource:@S
9convtranspose2d3_conv2d_transpose_readvariableop_resource:@@>
0convtranspose2d3_biasadd_readvariableop_resource:@)
bn3_readvariableop_resource:@+
bn3_readvariableop_1_resource:@:
,bn3_fusedbatchnormv3_readvariableop_resource:@<
.bn3_fusedbatchnormv3_readvariableop_1_resource:@S
9convtranspose2d4_conv2d_transpose_readvariableop_resource:@@>
0convtranspose2d4_biasadd_readvariableop_resource:@)
bn4_readvariableop_resource:@+
bn4_readvariableop_1_resource:@:
,bn4_fusedbatchnormv3_readvariableop_resource:@<
.bn4_fusedbatchnormv3_readvariableop_1_resource:@S
9convtranspose2d5_conv2d_transpose_readvariableop_resource: @>
0convtranspose2d5_biasadd_readvariableop_resource: )
bn5_readvariableop_resource: +
bn5_readvariableop_1_resource: :
,bn5_fusedbatchnormv3_readvariableop_resource: <
.bn5_fusedbatchnormv3_readvariableop_1_resource: S
9convtranspose2d6_conv2d_transpose_readvariableop_resource: >
0convtranspose2d6_biasadd_readvariableop_resource:
identity��bn1/AssignNewValue�bn1/AssignNewValue_1�#bn1/FusedBatchNormV3/ReadVariableOp�%bn1/FusedBatchNormV3/ReadVariableOp_1�bn1/ReadVariableOp�bn1/ReadVariableOp_1�bn2/AssignNewValue�bn2/AssignNewValue_1�#bn2/FusedBatchNormV3/ReadVariableOp�%bn2/FusedBatchNormV3/ReadVariableOp_1�bn2/ReadVariableOp�bn2/ReadVariableOp_1�bn3/AssignNewValue�bn3/AssignNewValue_1�#bn3/FusedBatchNormV3/ReadVariableOp�%bn3/FusedBatchNormV3/ReadVariableOp_1�bn3/ReadVariableOp�bn3/ReadVariableOp_1�bn4/AssignNewValue�bn4/AssignNewValue_1�#bn4/FusedBatchNormV3/ReadVariableOp�%bn4/FusedBatchNormV3/ReadVariableOp_1�bn4/ReadVariableOp�bn4/ReadVariableOp_1�bn5/AssignNewValue�bn5/AssignNewValue_1�#bn5/FusedBatchNormV3/ReadVariableOp�%bn5/FusedBatchNormV3/ReadVariableOp_1�bn5/ReadVariableOp�bn5/ReadVariableOp_1�'convTranspose2D1/BiasAdd/ReadVariableOp�0convTranspose2D1/conv2d_transpose/ReadVariableOp�'convTranspose2D2/BiasAdd/ReadVariableOp�0convTranspose2D2/conv2d_transpose/ReadVariableOp�'convTranspose2D3/BiasAdd/ReadVariableOp�0convTranspose2D3/conv2d_transpose/ReadVariableOp�'convTranspose2D4/BiasAdd/ReadVariableOp�0convTranspose2D4/conv2d_transpose/ReadVariableOp�'convTranspose2D5/BiasAdd/ReadVariableOp�0convTranspose2D5/conv2d_transpose/ReadVariableOp�'convTranspose2D6/BiasAdd/ReadVariableOp�0convTranspose2D6/conv2d_transpose/ReadVariableOp�dense1/BiasAdd/ReadVariableOp�dense1/MatMul/ReadVariableOp�
dense1/MatMul/ReadVariableOpReadVariableOp%dense1_matmul_readvariableop_resource*
_output_shapes
:	
�*
dtype0x
dense1/MatMulMatMulinputs$dense1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
dense1/BiasAdd/ReadVariableOpReadVariableOp&dense1_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
dense1/BiasAddBiasAdddense1/MatMul:product:0%dense1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������g
reshapeLayer/ShapeShapedense1/BiasAdd:output:0*
T0*
_output_shapes
::��j
 reshapeLayer/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: l
"reshapeLayer/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:l
"reshapeLayer/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
reshapeLayer/strided_sliceStridedSlicereshapeLayer/Shape:output:0)reshapeLayer/strided_slice/stack:output:0+reshapeLayer/strided_slice/stack_1:output:0+reshapeLayer/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask^
reshapeLayer/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :^
reshapeLayer/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :^
reshapeLayer/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B : �
reshapeLayer/Reshape/shapePack#reshapeLayer/strided_slice:output:0%reshapeLayer/Reshape/shape/1:output:0%reshapeLayer/Reshape/shape/2:output:0%reshapeLayer/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:�
reshapeLayer/ReshapeReshapedense1/BiasAdd:output:0#reshapeLayer/Reshape/shape:output:0*
T0*/
_output_shapes
:��������� q
convTranspose2D1/ShapeShapereshapeLayer/Reshape:output:0*
T0*
_output_shapes
::��n
$convTranspose2D1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: p
&convTranspose2D1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:p
&convTranspose2D1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
convTranspose2D1/strided_sliceStridedSliceconvTranspose2D1/Shape:output:0-convTranspose2D1/strided_slice/stack:output:0/convTranspose2D1/strided_slice/stack_1:output:0/convTranspose2D1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskZ
convTranspose2D1/stack/1Const*
_output_shapes
: *
dtype0*
value	B :Z
convTranspose2D1/stack/2Const*
_output_shapes
: *
dtype0*
value	B :Z
convTranspose2D1/stack/3Const*
_output_shapes
: *
dtype0*
value	B : �
convTranspose2D1/stackPack'convTranspose2D1/strided_slice:output:0!convTranspose2D1/stack/1:output:0!convTranspose2D1/stack/2:output:0!convTranspose2D1/stack/3:output:0*
N*
T0*
_output_shapes
:p
&convTranspose2D1/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: r
(convTranspose2D1/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:r
(convTranspose2D1/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
 convTranspose2D1/strided_slice_1StridedSliceconvTranspose2D1/stack:output:0/convTranspose2D1/strided_slice_1/stack:output:01convTranspose2D1/strided_slice_1/stack_1:output:01convTranspose2D1/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask�
0convTranspose2D1/conv2d_transpose/ReadVariableOpReadVariableOp9convtranspose2d1_conv2d_transpose_readvariableop_resource*&
_output_shapes
:  *
dtype0�
!convTranspose2D1/conv2d_transposeConv2DBackpropInputconvTranspose2D1/stack:output:08convTranspose2D1/conv2d_transpose/ReadVariableOp:value:0reshapeLayer/Reshape:output:0*
T0*/
_output_shapes
:��������� *
paddingSAME*
strides
�
'convTranspose2D1/BiasAdd/ReadVariableOpReadVariableOp0convtranspose2d1_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0�
convTranspose2D1/BiasAddBiasAdd*convTranspose2D1/conv2d_transpose:output:0/convTranspose2D1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:��������� j
bn1/ReadVariableOpReadVariableOpbn1_readvariableop_resource*
_output_shapes
: *
dtype0n
bn1/ReadVariableOp_1ReadVariableOpbn1_readvariableop_1_resource*
_output_shapes
: *
dtype0�
#bn1/FusedBatchNormV3/ReadVariableOpReadVariableOp,bn1_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype0�
%bn1/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp.bn1_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype0�
bn1/FusedBatchNormV3FusedBatchNormV3!convTranspose2D1/BiasAdd:output:0bn1/ReadVariableOp:value:0bn1/ReadVariableOp_1:value:0+bn1/FusedBatchNormV3/ReadVariableOp:value:0-bn1/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:��������� : : : : :*
epsilon%o�:*
exponential_avg_factor%
�#<�
bn1/AssignNewValueAssignVariableOp,bn1_fusedbatchnormv3_readvariableop_resource!bn1/FusedBatchNormV3:batch_mean:0$^bn1/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(�
bn1/AssignNewValue_1AssignVariableOp.bn1_fusedbatchnormv3_readvariableop_1_resource%bn1/FusedBatchNormV3:batch_variance:0&^bn1/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(x
lReLU1/LeakyRelu	LeakyRelubn1/FusedBatchNormV3:y:0*/
_output_shapes
:��������� *
alpha%���>r
convTranspose2D2/ShapeShapelReLU1/LeakyRelu:activations:0*
T0*
_output_shapes
::��n
$convTranspose2D2/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: p
&convTranspose2D2/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:p
&convTranspose2D2/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
convTranspose2D2/strided_sliceStridedSliceconvTranspose2D2/Shape:output:0-convTranspose2D2/strided_slice/stack:output:0/convTranspose2D2/strided_slice/stack_1:output:0/convTranspose2D2/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskZ
convTranspose2D2/stack/1Const*
_output_shapes
: *
dtype0*
value	B :Z
convTranspose2D2/stack/2Const*
_output_shapes
: *
dtype0*
value	B :Z
convTranspose2D2/stack/3Const*
_output_shapes
: *
dtype0*
value	B :@�
convTranspose2D2/stackPack'convTranspose2D2/strided_slice:output:0!convTranspose2D2/stack/1:output:0!convTranspose2D2/stack/2:output:0!convTranspose2D2/stack/3:output:0*
N*
T0*
_output_shapes
:p
&convTranspose2D2/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: r
(convTranspose2D2/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:r
(convTranspose2D2/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
 convTranspose2D2/strided_slice_1StridedSliceconvTranspose2D2/stack:output:0/convTranspose2D2/strided_slice_1/stack:output:01convTranspose2D2/strided_slice_1/stack_1:output:01convTranspose2D2/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask�
0convTranspose2D2/conv2d_transpose/ReadVariableOpReadVariableOp9convtranspose2d2_conv2d_transpose_readvariableop_resource*&
_output_shapes
:@ *
dtype0�
!convTranspose2D2/conv2d_transposeConv2DBackpropInputconvTranspose2D2/stack:output:08convTranspose2D2/conv2d_transpose/ReadVariableOp:value:0lReLU1/LeakyRelu:activations:0*
T0*/
_output_shapes
:���������@*
paddingSAME*
strides
�
'convTranspose2D2/BiasAdd/ReadVariableOpReadVariableOp0convtranspose2d2_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0�
convTranspose2D2/BiasAddBiasAdd*convTranspose2D2/conv2d_transpose:output:0/convTranspose2D2/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@j
bn2/ReadVariableOpReadVariableOpbn2_readvariableop_resource*
_output_shapes
:@*
dtype0n
bn2/ReadVariableOp_1ReadVariableOpbn2_readvariableop_1_resource*
_output_shapes
:@*
dtype0�
#bn2/FusedBatchNormV3/ReadVariableOpReadVariableOp,bn2_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype0�
%bn2/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp.bn2_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype0�
bn2/FusedBatchNormV3FusedBatchNormV3!convTranspose2D2/BiasAdd:output:0bn2/ReadVariableOp:value:0bn2/ReadVariableOp_1:value:0+bn2/FusedBatchNormV3/ReadVariableOp:value:0-bn2/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:���������@:@:@:@:@:*
epsilon%o�:*
exponential_avg_factor%
�#<�
bn2/AssignNewValueAssignVariableOp,bn2_fusedbatchnormv3_readvariableop_resource!bn2/FusedBatchNormV3:batch_mean:0$^bn2/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(�
bn2/AssignNewValue_1AssignVariableOp.bn2_fusedbatchnormv3_readvariableop_1_resource%bn2/FusedBatchNormV3:batch_variance:0&^bn2/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(x
lReLU2/LeakyRelu	LeakyRelubn2/FusedBatchNormV3:y:0*/
_output_shapes
:���������@*
alpha%���>r
convTranspose2D3/ShapeShapelReLU2/LeakyRelu:activations:0*
T0*
_output_shapes
::��n
$convTranspose2D3/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: p
&convTranspose2D3/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:p
&convTranspose2D3/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
convTranspose2D3/strided_sliceStridedSliceconvTranspose2D3/Shape:output:0-convTranspose2D3/strided_slice/stack:output:0/convTranspose2D3/strided_slice/stack_1:output:0/convTranspose2D3/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskZ
convTranspose2D3/stack/1Const*
_output_shapes
: *
dtype0*
value	B :Z
convTranspose2D3/stack/2Const*
_output_shapes
: *
dtype0*
value	B :Z
convTranspose2D3/stack/3Const*
_output_shapes
: *
dtype0*
value	B :@�
convTranspose2D3/stackPack'convTranspose2D3/strided_slice:output:0!convTranspose2D3/stack/1:output:0!convTranspose2D3/stack/2:output:0!convTranspose2D3/stack/3:output:0*
N*
T0*
_output_shapes
:p
&convTranspose2D3/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: r
(convTranspose2D3/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:r
(convTranspose2D3/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
 convTranspose2D3/strided_slice_1StridedSliceconvTranspose2D3/stack:output:0/convTranspose2D3/strided_slice_1/stack:output:01convTranspose2D3/strided_slice_1/stack_1:output:01convTranspose2D3/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask�
0convTranspose2D3/conv2d_transpose/ReadVariableOpReadVariableOp9convtranspose2d3_conv2d_transpose_readvariableop_resource*&
_output_shapes
:@@*
dtype0�
!convTranspose2D3/conv2d_transposeConv2DBackpropInputconvTranspose2D3/stack:output:08convTranspose2D3/conv2d_transpose/ReadVariableOp:value:0lReLU2/LeakyRelu:activations:0*
T0*/
_output_shapes
:���������@*
paddingSAME*
strides
�
'convTranspose2D3/BiasAdd/ReadVariableOpReadVariableOp0convtranspose2d3_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0�
convTranspose2D3/BiasAddBiasAdd*convTranspose2D3/conv2d_transpose:output:0/convTranspose2D3/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@j
bn3/ReadVariableOpReadVariableOpbn3_readvariableop_resource*
_output_shapes
:@*
dtype0n
bn3/ReadVariableOp_1ReadVariableOpbn3_readvariableop_1_resource*
_output_shapes
:@*
dtype0�
#bn3/FusedBatchNormV3/ReadVariableOpReadVariableOp,bn3_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype0�
%bn3/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp.bn3_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype0�
bn3/FusedBatchNormV3FusedBatchNormV3!convTranspose2D3/BiasAdd:output:0bn3/ReadVariableOp:value:0bn3/ReadVariableOp_1:value:0+bn3/FusedBatchNormV3/ReadVariableOp:value:0-bn3/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:���������@:@:@:@:@:*
epsilon%o�:*
exponential_avg_factor%
�#<�
bn3/AssignNewValueAssignVariableOp,bn3_fusedbatchnormv3_readvariableop_resource!bn3/FusedBatchNormV3:batch_mean:0$^bn3/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(�
bn3/AssignNewValue_1AssignVariableOp.bn3_fusedbatchnormv3_readvariableop_1_resource%bn3/FusedBatchNormV3:batch_variance:0&^bn3/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(x
lReLU3/LeakyRelu	LeakyRelubn3/FusedBatchNormV3:y:0*/
_output_shapes
:���������@*
alpha%���>r
convTranspose2D4/ShapeShapelReLU3/LeakyRelu:activations:0*
T0*
_output_shapes
::��n
$convTranspose2D4/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: p
&convTranspose2D4/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:p
&convTranspose2D4/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
convTranspose2D4/strided_sliceStridedSliceconvTranspose2D4/Shape:output:0-convTranspose2D4/strided_slice/stack:output:0/convTranspose2D4/strided_slice/stack_1:output:0/convTranspose2D4/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskZ
convTranspose2D4/stack/1Const*
_output_shapes
: *
dtype0*
value	B :Z
convTranspose2D4/stack/2Const*
_output_shapes
: *
dtype0*
value	B :Z
convTranspose2D4/stack/3Const*
_output_shapes
: *
dtype0*
value	B :@�
convTranspose2D4/stackPack'convTranspose2D4/strided_slice:output:0!convTranspose2D4/stack/1:output:0!convTranspose2D4/stack/2:output:0!convTranspose2D4/stack/3:output:0*
N*
T0*
_output_shapes
:p
&convTranspose2D4/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: r
(convTranspose2D4/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:r
(convTranspose2D4/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
 convTranspose2D4/strided_slice_1StridedSliceconvTranspose2D4/stack:output:0/convTranspose2D4/strided_slice_1/stack:output:01convTranspose2D4/strided_slice_1/stack_1:output:01convTranspose2D4/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask�
0convTranspose2D4/conv2d_transpose/ReadVariableOpReadVariableOp9convtranspose2d4_conv2d_transpose_readvariableop_resource*&
_output_shapes
:@@*
dtype0�
!convTranspose2D4/conv2d_transposeConv2DBackpropInputconvTranspose2D4/stack:output:08convTranspose2D4/conv2d_transpose/ReadVariableOp:value:0lReLU3/LeakyRelu:activations:0*
T0*/
_output_shapes
:���������@*
paddingSAME*
strides
�
'convTranspose2D4/BiasAdd/ReadVariableOpReadVariableOp0convtranspose2d4_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0�
convTranspose2D4/BiasAddBiasAdd*convTranspose2D4/conv2d_transpose:output:0/convTranspose2D4/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@j
bn4/ReadVariableOpReadVariableOpbn4_readvariableop_resource*
_output_shapes
:@*
dtype0n
bn4/ReadVariableOp_1ReadVariableOpbn4_readvariableop_1_resource*
_output_shapes
:@*
dtype0�
#bn4/FusedBatchNormV3/ReadVariableOpReadVariableOp,bn4_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype0�
%bn4/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp.bn4_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype0�
bn4/FusedBatchNormV3FusedBatchNormV3!convTranspose2D4/BiasAdd:output:0bn4/ReadVariableOp:value:0bn4/ReadVariableOp_1:value:0+bn4/FusedBatchNormV3/ReadVariableOp:value:0-bn4/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:���������@:@:@:@:@:*
epsilon%o�:*
exponential_avg_factor%
�#<�
bn4/AssignNewValueAssignVariableOp,bn4_fusedbatchnormv3_readvariableop_resource!bn4/FusedBatchNormV3:batch_mean:0$^bn4/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(�
bn4/AssignNewValue_1AssignVariableOp.bn4_fusedbatchnormv3_readvariableop_1_resource%bn4/FusedBatchNormV3:batch_variance:0&^bn4/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(x
lReLU4/LeakyRelu	LeakyRelubn4/FusedBatchNormV3:y:0*/
_output_shapes
:���������@*
alpha%���>r
convTranspose2D5/ShapeShapelReLU4/LeakyRelu:activations:0*
T0*
_output_shapes
::��n
$convTranspose2D5/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: p
&convTranspose2D5/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:p
&convTranspose2D5/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
convTranspose2D5/strided_sliceStridedSliceconvTranspose2D5/Shape:output:0-convTranspose2D5/strided_slice/stack:output:0/convTranspose2D5/strided_slice/stack_1:output:0/convTranspose2D5/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskZ
convTranspose2D5/stack/1Const*
_output_shapes
: *
dtype0*
value	B :Z
convTranspose2D5/stack/2Const*
_output_shapes
: *
dtype0*
value	B :Z
convTranspose2D5/stack/3Const*
_output_shapes
: *
dtype0*
value	B : �
convTranspose2D5/stackPack'convTranspose2D5/strided_slice:output:0!convTranspose2D5/stack/1:output:0!convTranspose2D5/stack/2:output:0!convTranspose2D5/stack/3:output:0*
N*
T0*
_output_shapes
:p
&convTranspose2D5/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: r
(convTranspose2D5/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:r
(convTranspose2D5/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
 convTranspose2D5/strided_slice_1StridedSliceconvTranspose2D5/stack:output:0/convTranspose2D5/strided_slice_1/stack:output:01convTranspose2D5/strided_slice_1/stack_1:output:01convTranspose2D5/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask�
0convTranspose2D5/conv2d_transpose/ReadVariableOpReadVariableOp9convtranspose2d5_conv2d_transpose_readvariableop_resource*&
_output_shapes
: @*
dtype0�
!convTranspose2D5/conv2d_transposeConv2DBackpropInputconvTranspose2D5/stack:output:08convTranspose2D5/conv2d_transpose/ReadVariableOp:value:0lReLU4/LeakyRelu:activations:0*
T0*/
_output_shapes
:��������� *
paddingSAME*
strides
�
'convTranspose2D5/BiasAdd/ReadVariableOpReadVariableOp0convtranspose2d5_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0�
convTranspose2D5/BiasAddBiasAdd*convTranspose2D5/conv2d_transpose:output:0/convTranspose2D5/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:��������� j
bn5/ReadVariableOpReadVariableOpbn5_readvariableop_resource*
_output_shapes
: *
dtype0n
bn5/ReadVariableOp_1ReadVariableOpbn5_readvariableop_1_resource*
_output_shapes
: *
dtype0�
#bn5/FusedBatchNormV3/ReadVariableOpReadVariableOp,bn5_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype0�
%bn5/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp.bn5_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype0�
bn5/FusedBatchNormV3FusedBatchNormV3!convTranspose2D5/BiasAdd:output:0bn5/ReadVariableOp:value:0bn5/ReadVariableOp_1:value:0+bn5/FusedBatchNormV3/ReadVariableOp:value:0-bn5/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:��������� : : : : :*
epsilon%o�:*
exponential_avg_factor%
�#<�
bn5/AssignNewValueAssignVariableOp,bn5_fusedbatchnormv3_readvariableop_resource!bn5/FusedBatchNormV3:batch_mean:0$^bn5/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(�
bn5/AssignNewValue_1AssignVariableOp.bn5_fusedbatchnormv3_readvariableop_1_resource%bn5/FusedBatchNormV3:batch_variance:0&^bn5/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(x
lReLU5/LeakyRelu	LeakyRelubn5/FusedBatchNormV3:y:0*/
_output_shapes
:��������� *
alpha%���>r
convTranspose2D6/ShapeShapelReLU5/LeakyRelu:activations:0*
T0*
_output_shapes
::��n
$convTranspose2D6/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: p
&convTranspose2D6/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:p
&convTranspose2D6/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
convTranspose2D6/strided_sliceStridedSliceconvTranspose2D6/Shape:output:0-convTranspose2D6/strided_slice/stack:output:0/convTranspose2D6/strided_slice/stack_1:output:0/convTranspose2D6/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskZ
convTranspose2D6/stack/1Const*
_output_shapes
: *
dtype0*
value	B :Z
convTranspose2D6/stack/2Const*
_output_shapes
: *
dtype0*
value	B :Z
convTranspose2D6/stack/3Const*
_output_shapes
: *
dtype0*
value	B :�
convTranspose2D6/stackPack'convTranspose2D6/strided_slice:output:0!convTranspose2D6/stack/1:output:0!convTranspose2D6/stack/2:output:0!convTranspose2D6/stack/3:output:0*
N*
T0*
_output_shapes
:p
&convTranspose2D6/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: r
(convTranspose2D6/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:r
(convTranspose2D6/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
 convTranspose2D6/strided_slice_1StridedSliceconvTranspose2D6/stack:output:0/convTranspose2D6/strided_slice_1/stack:output:01convTranspose2D6/strided_slice_1/stack_1:output:01convTranspose2D6/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask�
0convTranspose2D6/conv2d_transpose/ReadVariableOpReadVariableOp9convtranspose2d6_conv2d_transpose_readvariableop_resource*&
_output_shapes
: *
dtype0�
!convTranspose2D6/conv2d_transposeConv2DBackpropInputconvTranspose2D6/stack:output:08convTranspose2D6/conv2d_transpose/ReadVariableOp:value:0lReLU5/LeakyRelu:activations:0*
T0*/
_output_shapes
:���������*
paddingSAME*
strides
�
'convTranspose2D6/BiasAdd/ReadVariableOpReadVariableOp0convtranspose2d6_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
convTranspose2D6/BiasAddBiasAdd*convTranspose2D6/conv2d_transpose:output:0/convTranspose2D6/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:����������
convTranspose2D6/SigmoidSigmoid!convTranspose2D6/BiasAdd:output:0*
T0*/
_output_shapes
:���������s
IdentityIdentityconvTranspose2D6/Sigmoid:y:0^NoOp*
T0*/
_output_shapes
:����������
NoOpNoOp^bn1/AssignNewValue^bn1/AssignNewValue_1$^bn1/FusedBatchNormV3/ReadVariableOp&^bn1/FusedBatchNormV3/ReadVariableOp_1^bn1/ReadVariableOp^bn1/ReadVariableOp_1^bn2/AssignNewValue^bn2/AssignNewValue_1$^bn2/FusedBatchNormV3/ReadVariableOp&^bn2/FusedBatchNormV3/ReadVariableOp_1^bn2/ReadVariableOp^bn2/ReadVariableOp_1^bn3/AssignNewValue^bn3/AssignNewValue_1$^bn3/FusedBatchNormV3/ReadVariableOp&^bn3/FusedBatchNormV3/ReadVariableOp_1^bn3/ReadVariableOp^bn3/ReadVariableOp_1^bn4/AssignNewValue^bn4/AssignNewValue_1$^bn4/FusedBatchNormV3/ReadVariableOp&^bn4/FusedBatchNormV3/ReadVariableOp_1^bn4/ReadVariableOp^bn4/ReadVariableOp_1^bn5/AssignNewValue^bn5/AssignNewValue_1$^bn5/FusedBatchNormV3/ReadVariableOp&^bn5/FusedBatchNormV3/ReadVariableOp_1^bn5/ReadVariableOp^bn5/ReadVariableOp_1(^convTranspose2D1/BiasAdd/ReadVariableOp1^convTranspose2D1/conv2d_transpose/ReadVariableOp(^convTranspose2D2/BiasAdd/ReadVariableOp1^convTranspose2D2/conv2d_transpose/ReadVariableOp(^convTranspose2D3/BiasAdd/ReadVariableOp1^convTranspose2D3/conv2d_transpose/ReadVariableOp(^convTranspose2D4/BiasAdd/ReadVariableOp1^convTranspose2D4/conv2d_transpose/ReadVariableOp(^convTranspose2D5/BiasAdd/ReadVariableOp1^convTranspose2D5/conv2d_transpose/ReadVariableOp(^convTranspose2D6/BiasAdd/ReadVariableOp1^convTranspose2D6/conv2d_transpose/ReadVariableOp^dense1/BiasAdd/ReadVariableOp^dense1/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*j
_input_shapesY
W:���������
: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2(
bn1/AssignNewValuebn1/AssignNewValue2,
bn1/AssignNewValue_1bn1/AssignNewValue_12J
#bn1/FusedBatchNormV3/ReadVariableOp#bn1/FusedBatchNormV3/ReadVariableOp2N
%bn1/FusedBatchNormV3/ReadVariableOp_1%bn1/FusedBatchNormV3/ReadVariableOp_12(
bn1/ReadVariableOpbn1/ReadVariableOp2,
bn1/ReadVariableOp_1bn1/ReadVariableOp_12(
bn2/AssignNewValuebn2/AssignNewValue2,
bn2/AssignNewValue_1bn2/AssignNewValue_12J
#bn2/FusedBatchNormV3/ReadVariableOp#bn2/FusedBatchNormV3/ReadVariableOp2N
%bn2/FusedBatchNormV3/ReadVariableOp_1%bn2/FusedBatchNormV3/ReadVariableOp_12(
bn2/ReadVariableOpbn2/ReadVariableOp2,
bn2/ReadVariableOp_1bn2/ReadVariableOp_12(
bn3/AssignNewValuebn3/AssignNewValue2,
bn3/AssignNewValue_1bn3/AssignNewValue_12J
#bn3/FusedBatchNormV3/ReadVariableOp#bn3/FusedBatchNormV3/ReadVariableOp2N
%bn3/FusedBatchNormV3/ReadVariableOp_1%bn3/FusedBatchNormV3/ReadVariableOp_12(
bn3/ReadVariableOpbn3/ReadVariableOp2,
bn3/ReadVariableOp_1bn3/ReadVariableOp_12(
bn4/AssignNewValuebn4/AssignNewValue2,
bn4/AssignNewValue_1bn4/AssignNewValue_12J
#bn4/FusedBatchNormV3/ReadVariableOp#bn4/FusedBatchNormV3/ReadVariableOp2N
%bn4/FusedBatchNormV3/ReadVariableOp_1%bn4/FusedBatchNormV3/ReadVariableOp_12(
bn4/ReadVariableOpbn4/ReadVariableOp2,
bn4/ReadVariableOp_1bn4/ReadVariableOp_12(
bn5/AssignNewValuebn5/AssignNewValue2,
bn5/AssignNewValue_1bn5/AssignNewValue_12J
#bn5/FusedBatchNormV3/ReadVariableOp#bn5/FusedBatchNormV3/ReadVariableOp2N
%bn5/FusedBatchNormV3/ReadVariableOp_1%bn5/FusedBatchNormV3/ReadVariableOp_12(
bn5/ReadVariableOpbn5/ReadVariableOp2,
bn5/ReadVariableOp_1bn5/ReadVariableOp_12R
'convTranspose2D1/BiasAdd/ReadVariableOp'convTranspose2D1/BiasAdd/ReadVariableOp2d
0convTranspose2D1/conv2d_transpose/ReadVariableOp0convTranspose2D1/conv2d_transpose/ReadVariableOp2R
'convTranspose2D2/BiasAdd/ReadVariableOp'convTranspose2D2/BiasAdd/ReadVariableOp2d
0convTranspose2D2/conv2d_transpose/ReadVariableOp0convTranspose2D2/conv2d_transpose/ReadVariableOp2R
'convTranspose2D3/BiasAdd/ReadVariableOp'convTranspose2D3/BiasAdd/ReadVariableOp2d
0convTranspose2D3/conv2d_transpose/ReadVariableOp0convTranspose2D3/conv2d_transpose/ReadVariableOp2R
'convTranspose2D4/BiasAdd/ReadVariableOp'convTranspose2D4/BiasAdd/ReadVariableOp2d
0convTranspose2D4/conv2d_transpose/ReadVariableOp0convTranspose2D4/conv2d_transpose/ReadVariableOp2R
'convTranspose2D5/BiasAdd/ReadVariableOp'convTranspose2D5/BiasAdd/ReadVariableOp2d
0convTranspose2D5/conv2d_transpose/ReadVariableOp0convTranspose2D5/conv2d_transpose/ReadVariableOp2R
'convTranspose2D6/BiasAdd/ReadVariableOp'convTranspose2D6/BiasAdd/ReadVariableOp2d
0convTranspose2D6/conv2d_transpose/ReadVariableOp0convTranspose2D6/conv2d_transpose/ReadVariableOp2>
dense1/BiasAdd/ReadVariableOpdense1/BiasAdd/ReadVariableOp2<
dense1/MatMul/ReadVariableOpdense1/MatMul/ReadVariableOp:O K
'
_output_shapes
:���������

 
_user_specified_nameinputs
�
�
0__inference_convTranspose2D3_layer_call_fn_17119

inputs!
unknown:@@
	unknown_0:@
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+���������������������������@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *T
fORM
K__inference_convTranspose2D3_layer_call_and_return_conditional_losses_15112�
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+���������������������������@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:+���������������������������@: : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+���������������������������@
 
_user_specified_nameinputs
�!
�
K__inference_convTranspose2D6_layer_call_and_return_conditional_losses_15437

inputsB
(conv2d_transpose_readvariableop_resource: -
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�conv2d_transpose/ReadVariableOpI
ShapeShapeinputs*
T0*
_output_shapes
::��]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_1StridedSliceShape:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_2StridedSliceShape:output:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskG
mul/yConst*
_output_shapes
: *
dtype0*
value	B :U
mulMulstrided_slice_1:output:0mul/y:output:0*
T0*
_output_shapes
: I
mul_1/yConst*
_output_shapes
: *
dtype0*
value	B :Y
mul_1Mulstrided_slice_2:output:0mul_1/y:output:0*
T0*
_output_shapes
: I
stack/3Const*
_output_shapes
: *
dtype0*
value	B :y
stackPackstrided_slice:output:0mul:z:0	mul_1:z:0stack/3:output:0*
N*
T0*
_output_shapes
:_
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_3StridedSlicestack:output:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask�
conv2d_transpose/ReadVariableOpReadVariableOp(conv2d_transpose_readvariableop_resource*&
_output_shapes
: *
dtype0�
conv2d_transposeConv2DBackpropInputstack:output:0'conv2d_transpose/ReadVariableOp:value:0inputs*
T0*A
_output_shapes/
-:+���������������������������*
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
BiasAddBiasAddconv2d_transpose:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+���������������������������p
SigmoidSigmoidBiasAdd:output:0*
T0*A
_output_shapes/
-:+���������������������������t
IdentityIdentitySigmoid:y:0^NoOp*
T0*A
_output_shapes/
-:+����������������������������
NoOpNoOp^BiasAdd/ReadVariableOp ^conv2d_transpose/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:+��������������������������� : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2B
conv2d_transpose/ReadVariableOpconv2d_transpose/ReadVariableOp:i e
A
_output_shapes/
-:+��������������������������� 
 
_user_specified_nameinputs
�
�
#__inference_bn3_layer_call_fn_17178

inputs
unknown:@
	unknown_0:@
	unknown_1:@
	unknown_2:@
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+���������������������������@*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *G
fBR@
>__inference_bn3_layer_call_and_return_conditional_losses_15159�
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+���������������������������@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+���������������������������@: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+���������������������������@
 
_user_specified_nameinputs
�
�
'__inference_decoder_layer_call_fn_16351

inputs
unknown:	
�
	unknown_0:	�#
	unknown_1:  
	unknown_2: 
	unknown_3: 
	unknown_4: 
	unknown_5: 
	unknown_6: #
	unknown_7:@ 
	unknown_8:@
	unknown_9:@

unknown_10:@

unknown_11:@

unknown_12:@$

unknown_13:@@

unknown_14:@

unknown_15:@

unknown_16:@

unknown_17:@

unknown_18:@$

unknown_19:@@

unknown_20:@

unknown_21:@

unknown_22:@

unknown_23:@

unknown_24:@$

unknown_25: @

unknown_26: 

unknown_27: 

unknown_28: 

unknown_29: 

unknown_30: $

unknown_31: 

unknown_32:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28
unknown_29
unknown_30
unknown_31
unknown_32*.
Tin'
%2#*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������*:
_read_only_resource_inputs
	
!"*-
config_proto

CPU

GPU 2J 8� *K
fFRD
B__inference_decoder_layer_call_and_return_conditional_losses_15777w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:���������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*j
_input_shapesY
W:���������
: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:���������

 
_user_specified_nameinputs
�
�
#__inference_signature_wrapper_16278

inputlayer
unknown:	
�
	unknown_0:	�#
	unknown_1:  
	unknown_2: 
	unknown_3: 
	unknown_4: 
	unknown_5: 
	unknown_6: #
	unknown_7:@ 
	unknown_8:@
	unknown_9:@

unknown_10:@

unknown_11:@

unknown_12:@$

unknown_13:@@

unknown_14:@

unknown_15:@

unknown_16:@

unknown_17:@

unknown_18:@$

unknown_19:@@

unknown_20:@

unknown_21:@

unknown_22:@

unknown_23:@

unknown_24:@$

unknown_25: @

unknown_26: 

unknown_27: 

unknown_28: 

unknown_29: 

unknown_30: $

unknown_31: 

unknown_32:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCall
inputlayerunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28
unknown_29
unknown_30
unknown_31
unknown_32*.
Tin'
%2#*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������*D
_read_only_resource_inputs&
$"	
 !"*-
config_proto

CPU

GPU 2J 8� *)
f$R"
 __inference__wrapped_model_14862w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:���������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*j
_input_shapesY
W:���������
: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
'
_output_shapes
:���������

$
_user_specified_name
inputLayer
�
]
A__inference_lReLU4_layer_call_and_return_conditional_losses_15565

inputs
identity_
	LeakyRelu	LeakyReluinputs*/
_output_shapes
:���������@*
alpha%���>g
IdentityIdentityLeakyRelu:activations:0*
T0*/
_output_shapes
:���������@"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������@:W S
/
_output_shapes
:���������@
 
_user_specified_nameinputs
� 
�
K__inference_convTranspose2D3_layer_call_and_return_conditional_losses_17152

inputsB
(conv2d_transpose_readvariableop_resource:@@-
biasadd_readvariableop_resource:@
identity��BiasAdd/ReadVariableOp�conv2d_transpose/ReadVariableOpI
ShapeShapeinputs*
T0*
_output_shapes
::��]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_1StridedSliceShape:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_2StridedSliceShape:output:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskG
mul/yConst*
_output_shapes
: *
dtype0*
value	B :U
mulMulstrided_slice_1:output:0mul/y:output:0*
T0*
_output_shapes
: I
mul_1/yConst*
_output_shapes
: *
dtype0*
value	B :Y
mul_1Mulstrided_slice_2:output:0mul_1/y:output:0*
T0*
_output_shapes
: I
stack/3Const*
_output_shapes
: *
dtype0*
value	B :@y
stackPackstrided_slice:output:0mul:z:0	mul_1:z:0stack/3:output:0*
N*
T0*
_output_shapes
:_
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_3StridedSlicestack:output:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask�
conv2d_transpose/ReadVariableOpReadVariableOp(conv2d_transpose_readvariableop_resource*&
_output_shapes
:@@*
dtype0�
conv2d_transposeConv2DBackpropInputstack:output:0'conv2d_transpose/ReadVariableOp:value:0inputs*
T0*A
_output_shapes/
-:+���������������������������@*
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0�
BiasAddBiasAddconv2d_transpose:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+���������������������������@y
IdentityIdentityBiasAdd:output:0^NoOp*
T0*A
_output_shapes/
-:+���������������������������@�
NoOpNoOp^BiasAdd/ReadVariableOp ^conv2d_transpose/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:+���������������������������@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2B
conv2d_transpose/ReadVariableOpconv2d_transpose/ReadVariableOp:i e
A
_output_shapes/
-:+���������������������������@
 
_user_specified_nameinputs
�
�
>__inference_bn3_layer_call_and_return_conditional_losses_17214

inputs%
readvariableop_resource:@'
readvariableop_1_resource:@6
(fusedbatchnormv3_readvariableop_resource:@8
*fusedbatchnormv3_readvariableop_1_resource:@
identity��FusedBatchNormV3/ReadVariableOp�!FusedBatchNormV3/ReadVariableOp_1�ReadVariableOp�ReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype0�
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype0�
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype0�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+���������������������������@:@:@:@:@:*
epsilon%o�:*
is_training( }
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+���������������������������@�
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+���������������������������@: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+���������������������������@
 
_user_specified_nameinputs
�
�
>__inference_bn1_layer_call_and_return_conditional_losses_14925

inputs%
readvariableop_resource: '
readvariableop_1_resource: 6
(fusedbatchnormv3_readvariableop_resource: 8
*fusedbatchnormv3_readvariableop_1_resource: 
identity��AssignNewValue�AssignNewValue_1�FusedBatchNormV3/ReadVariableOp�!FusedBatchNormV3/ReadVariableOp_1�ReadVariableOp�ReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype0�
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype0�
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype0�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+��������������������������� : : : : :*
epsilon%o�:*
exponential_avg_factor%
�#<�
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(�
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(}
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+��������������������������� �
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+��������������������������� : : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+��������������������������� 
 
_user_specified_nameinputs
�
�
>__inference_bn4_layer_call_and_return_conditional_losses_17310

inputs%
readvariableop_resource:@'
readvariableop_1_resource:@6
(fusedbatchnormv3_readvariableop_resource:@8
*fusedbatchnormv3_readvariableop_1_resource:@
identity��AssignNewValue�AssignNewValue_1�FusedBatchNormV3/ReadVariableOp�!FusedBatchNormV3/ReadVariableOp_1�ReadVariableOp�ReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype0�
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype0�
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype0�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+���������������������������@:@:@:@:@:*
epsilon%o�:*
exponential_avg_factor%
�#<�
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(�
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(}
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+���������������������������@�
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+���������������������������@: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+���������������������������@
 
_user_specified_nameinputs
�
]
A__inference_lReLU5_layer_call_and_return_conditional_losses_17452

inputs
identity_
	LeakyRelu	LeakyReluinputs*/
_output_shapes
:��������� *
alpha%���>g
IdentityIdentityLeakyRelu:activations:0*
T0*/
_output_shapes
:��������� "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:��������� :W S
/
_output_shapes
:��������� 
 
_user_specified_nameinputs
�
B
&__inference_lReLU3_layer_call_fn_17219

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *J
fERC
A__inference_lReLU3_layer_call_and_return_conditional_losses_15544h
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:���������@"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������@:W S
/
_output_shapes
:���������@
 
_user_specified_nameinputs
�
�
#__inference_bn5_layer_call_fn_17406

inputs
unknown: 
	unknown_0: 
	unknown_1: 
	unknown_2: 
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+��������������������������� *&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *G
fBR@
>__inference_bn5_layer_call_and_return_conditional_losses_15375�
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+��������������������������� `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+��������������������������� : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+��������������������������� 
 
_user_specified_nameinputs
�
B
&__inference_lReLU2_layer_call_fn_17105

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *J
fERC
A__inference_lReLU2_layer_call_and_return_conditional_losses_15523h
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:���������@"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������@:W S
/
_output_shapes
:���������@
 
_user_specified_nameinputs
� 
�
K__inference_convTranspose2D5_layer_call_and_return_conditional_losses_15328

inputsB
(conv2d_transpose_readvariableop_resource: @-
biasadd_readvariableop_resource: 
identity��BiasAdd/ReadVariableOp�conv2d_transpose/ReadVariableOpI
ShapeShapeinputs*
T0*
_output_shapes
::��]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_1StridedSliceShape:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_2StridedSliceShape:output:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskG
mul/yConst*
_output_shapes
: *
dtype0*
value	B :U
mulMulstrided_slice_1:output:0mul/y:output:0*
T0*
_output_shapes
: I
mul_1/yConst*
_output_shapes
: *
dtype0*
value	B :Y
mul_1Mulstrided_slice_2:output:0mul_1/y:output:0*
T0*
_output_shapes
: I
stack/3Const*
_output_shapes
: *
dtype0*
value	B : y
stackPackstrided_slice:output:0mul:z:0	mul_1:z:0stack/3:output:0*
N*
T0*
_output_shapes
:_
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_3StridedSlicestack:output:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask�
conv2d_transpose/ReadVariableOpReadVariableOp(conv2d_transpose_readvariableop_resource*&
_output_shapes
: @*
dtype0�
conv2d_transposeConv2DBackpropInputstack:output:0'conv2d_transpose/ReadVariableOp:value:0inputs*
T0*A
_output_shapes/
-:+��������������������������� *
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype0�
BiasAddBiasAddconv2d_transpose:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+��������������������������� y
IdentityIdentityBiasAdd:output:0^NoOp*
T0*A
_output_shapes/
-:+��������������������������� �
NoOpNoOp^BiasAdd/ReadVariableOp ^conv2d_transpose/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:+���������������������������@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2B
conv2d_transpose/ReadVariableOpconv2d_transpose/ReadVariableOp:i e
A
_output_shapes/
-:+���������������������������@
 
_user_specified_nameinputs
�
�
#__inference_bn2_layer_call_fn_17064

inputs
unknown:@
	unknown_0:@
	unknown_1:@
	unknown_2:@
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+���������������������������@*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *G
fBR@
>__inference_bn2_layer_call_and_return_conditional_losses_15051�
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+���������������������������@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+���������������������������@: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+���������������������������@
 
_user_specified_nameinputs
� 
�
K__inference_convTranspose2D4_layer_call_and_return_conditional_losses_15220

inputsB
(conv2d_transpose_readvariableop_resource:@@-
biasadd_readvariableop_resource:@
identity��BiasAdd/ReadVariableOp�conv2d_transpose/ReadVariableOpI
ShapeShapeinputs*
T0*
_output_shapes
::��]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_1StridedSliceShape:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_2StridedSliceShape:output:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskG
mul/yConst*
_output_shapes
: *
dtype0*
value	B :U
mulMulstrided_slice_1:output:0mul/y:output:0*
T0*
_output_shapes
: I
mul_1/yConst*
_output_shapes
: *
dtype0*
value	B :Y
mul_1Mulstrided_slice_2:output:0mul_1/y:output:0*
T0*
_output_shapes
: I
stack/3Const*
_output_shapes
: *
dtype0*
value	B :@y
stackPackstrided_slice:output:0mul:z:0	mul_1:z:0stack/3:output:0*
N*
T0*
_output_shapes
:_
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_3StridedSlicestack:output:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask�
conv2d_transpose/ReadVariableOpReadVariableOp(conv2d_transpose_readvariableop_resource*&
_output_shapes
:@@*
dtype0�
conv2d_transposeConv2DBackpropInputstack:output:0'conv2d_transpose/ReadVariableOp:value:0inputs*
T0*A
_output_shapes/
-:+���������������������������@*
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0�
BiasAddBiasAddconv2d_transpose:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+���������������������������@y
IdentityIdentityBiasAdd:output:0^NoOp*
T0*A
_output_shapes/
-:+���������������������������@�
NoOpNoOp^BiasAdd/ReadVariableOp ^conv2d_transpose/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:+���������������������������@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2B
conv2d_transpose/ReadVariableOpconv2d_transpose/ReadVariableOp:i e
A
_output_shapes/
-:+���������������������������@
 
_user_specified_nameinputs
�
�
>__inference_bn2_layer_call_and_return_conditional_losses_15033

inputs%
readvariableop_resource:@'
readvariableop_1_resource:@6
(fusedbatchnormv3_readvariableop_resource:@8
*fusedbatchnormv3_readvariableop_1_resource:@
identity��AssignNewValue�AssignNewValue_1�FusedBatchNormV3/ReadVariableOp�!FusedBatchNormV3/ReadVariableOp_1�ReadVariableOp�ReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype0�
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype0�
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype0�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+���������������������������@:@:@:@:@:*
epsilon%o�:*
exponential_avg_factor%
�#<�
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(�
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(}
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+���������������������������@�
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+���������������������������@: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+���������������������������@
 
_user_specified_nameinputs
�
�
>__inference_bn4_layer_call_and_return_conditional_losses_15267

inputs%
readvariableop_resource:@'
readvariableop_1_resource:@6
(fusedbatchnormv3_readvariableop_resource:@8
*fusedbatchnormv3_readvariableop_1_resource:@
identity��FusedBatchNormV3/ReadVariableOp�!FusedBatchNormV3/ReadVariableOp_1�ReadVariableOp�ReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype0�
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype0�
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype0�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+���������������������������@:@:@:@:@:*
epsilon%o�:*
is_training( }
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+���������������������������@�
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+���������������������������@: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+���������������������������@
 
_user_specified_nameinputs
�
B
&__inference_lReLU4_layer_call_fn_17333

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *J
fERC
A__inference_lReLU4_layer_call_and_return_conditional_losses_15565h
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:���������@"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������@:W S
/
_output_shapes
:���������@
 
_user_specified_nameinputs
�
�
#__inference_bn1_layer_call_fn_16937

inputs
unknown: 
	unknown_0: 
	unknown_1: 
	unknown_2: 
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+��������������������������� *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *G
fBR@
>__inference_bn1_layer_call_and_return_conditional_losses_14925�
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+��������������������������� `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+��������������������������� : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+��������������������������� 
 
_user_specified_nameinputs
�
]
A__inference_lReLU3_layer_call_and_return_conditional_losses_15544

inputs
identity_
	LeakyRelu	LeakyReluinputs*/
_output_shapes
:���������@*
alpha%���>g
IdentityIdentityLeakyRelu:activations:0*
T0*/
_output_shapes
:���������@"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������@:W S
/
_output_shapes
:���������@
 
_user_specified_nameinputs
�!
�
K__inference_convTranspose2D6_layer_call_and_return_conditional_losses_17495

inputsB
(conv2d_transpose_readvariableop_resource: -
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�conv2d_transpose/ReadVariableOpI
ShapeShapeinputs*
T0*
_output_shapes
::��]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_1StridedSliceShape:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_2StridedSliceShape:output:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskG
mul/yConst*
_output_shapes
: *
dtype0*
value	B :U
mulMulstrided_slice_1:output:0mul/y:output:0*
T0*
_output_shapes
: I
mul_1/yConst*
_output_shapes
: *
dtype0*
value	B :Y
mul_1Mulstrided_slice_2:output:0mul_1/y:output:0*
T0*
_output_shapes
: I
stack/3Const*
_output_shapes
: *
dtype0*
value	B :y
stackPackstrided_slice:output:0mul:z:0	mul_1:z:0stack/3:output:0*
N*
T0*
_output_shapes
:_
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_3StridedSlicestack:output:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask�
conv2d_transpose/ReadVariableOpReadVariableOp(conv2d_transpose_readvariableop_resource*&
_output_shapes
: *
dtype0�
conv2d_transposeConv2DBackpropInputstack:output:0'conv2d_transpose/ReadVariableOp:value:0inputs*
T0*A
_output_shapes/
-:+���������������������������*
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
BiasAddBiasAddconv2d_transpose:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+���������������������������p
SigmoidSigmoidBiasAdd:output:0*
T0*A
_output_shapes/
-:+���������������������������t
IdentityIdentitySigmoid:y:0^NoOp*
T0*A
_output_shapes/
-:+����������������������������
NoOpNoOp^BiasAdd/ReadVariableOp ^conv2d_transpose/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:+��������������������������� : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2B
conv2d_transpose/ReadVariableOpconv2d_transpose/ReadVariableOp:i e
A
_output_shapes/
-:+��������������������������� 
 
_user_specified_nameinputs
�
]
A__inference_lReLU4_layer_call_and_return_conditional_losses_17338

inputs
identity_
	LeakyRelu	LeakyReluinputs*/
_output_shapes
:���������@*
alpha%���>g
IdentityIdentityLeakyRelu:activations:0*
T0*/
_output_shapes
:���������@"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������@:W S
/
_output_shapes
:���������@
 
_user_specified_nameinputs
�
�
>__inference_bn5_layer_call_and_return_conditional_losses_17424

inputs%
readvariableop_resource: '
readvariableop_1_resource: 6
(fusedbatchnormv3_readvariableop_resource: 8
*fusedbatchnormv3_readvariableop_1_resource: 
identity��AssignNewValue�AssignNewValue_1�FusedBatchNormV3/ReadVariableOp�!FusedBatchNormV3/ReadVariableOp_1�ReadVariableOp�ReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype0�
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype0�
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype0�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+��������������������������� : : : : :*
epsilon%o�:*
exponential_avg_factor%
�#<�
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(�
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(}
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+��������������������������� �
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+��������������������������� : : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+��������������������������� 
 
_user_specified_nameinputs
�
�
&__inference_dense1_layer_call_fn_16853

inputs
unknown:	
�
	unknown_0:	�
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *J
fERC
A__inference_dense1_layer_call_and_return_conditional_losses_15461p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:����������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������
: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:���������

 
_user_specified_nameinputs
�	
�
A__inference_dense1_layer_call_and_return_conditional_losses_15461

inputs1
matmul_readvariableop_resource:	
�.
biasadd_readvariableop_resource:	�
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	
�*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������`
IdentityIdentityBiasAdd:output:0^NoOp*
T0*(
_output_shapes
:����������w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������
: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:���������

 
_user_specified_nameinputs
�
�
#__inference_bn1_layer_call_fn_16950

inputs
unknown: 
	unknown_0: 
	unknown_1: 
	unknown_2: 
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+��������������������������� *&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *G
fBR@
>__inference_bn1_layer_call_and_return_conditional_losses_14943�
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+��������������������������� `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+��������������������������� : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+��������������������������� 
 
_user_specified_nameinputs
�
�
>__inference_bn2_layer_call_and_return_conditional_losses_17100

inputs%
readvariableop_resource:@'
readvariableop_1_resource:@6
(fusedbatchnormv3_readvariableop_resource:@8
*fusedbatchnormv3_readvariableop_1_resource:@
identity��FusedBatchNormV3/ReadVariableOp�!FusedBatchNormV3/ReadVariableOp_1�ReadVariableOp�ReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype0�
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype0�
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype0�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+���������������������������@:@:@:@:@:*
epsilon%o�:*
is_training( }
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+���������������������������@�
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+���������������������������@: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+���������������������������@
 
_user_specified_nameinputs
�S
�
B__inference_decoder_layer_call_and_return_conditional_losses_15684

inputlayer
dense1_15597:	
�
dense1_15599:	�0
convtranspose2d1_15603:  $
convtranspose2d1_15605: 
	bn1_15608: 
	bn1_15610: 
	bn1_15612: 
	bn1_15614: 0
convtranspose2d2_15618:@ $
convtranspose2d2_15620:@
	bn2_15623:@
	bn2_15625:@
	bn2_15627:@
	bn2_15629:@0
convtranspose2d3_15633:@@$
convtranspose2d3_15635:@
	bn3_15638:@
	bn3_15640:@
	bn3_15642:@
	bn3_15644:@0
convtranspose2d4_15648:@@$
convtranspose2d4_15650:@
	bn4_15653:@
	bn4_15655:@
	bn4_15657:@
	bn4_15659:@0
convtranspose2d5_15663: @$
convtranspose2d5_15665: 
	bn5_15668: 
	bn5_15670: 
	bn5_15672: 
	bn5_15674: 0
convtranspose2d6_15678: $
convtranspose2d6_15680:
identity��bn1/StatefulPartitionedCall�bn2/StatefulPartitionedCall�bn3/StatefulPartitionedCall�bn4/StatefulPartitionedCall�bn5/StatefulPartitionedCall�(convTranspose2D1/StatefulPartitionedCall�(convTranspose2D2/StatefulPartitionedCall�(convTranspose2D3/StatefulPartitionedCall�(convTranspose2D4/StatefulPartitionedCall�(convTranspose2D5/StatefulPartitionedCall�(convTranspose2D6/StatefulPartitionedCall�dense1/StatefulPartitionedCall�
dense1/StatefulPartitionedCallStatefulPartitionedCall
inputlayerdense1_15597dense1_15599*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *J
fERC
A__inference_dense1_layer_call_and_return_conditional_losses_15461�
reshapeLayer/PartitionedCallPartitionedCall'dense1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:��������� * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *P
fKRI
G__inference_reshapeLayer_layer_call_and_return_conditional_losses_15481�
(convTranspose2D1/StatefulPartitionedCallStatefulPartitionedCall%reshapeLayer/PartitionedCall:output:0convtranspose2d1_15603convtranspose2d1_15605*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:��������� *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *T
fORM
K__inference_convTranspose2D1_layer_call_and_return_conditional_losses_14896�
bn1/StatefulPartitionedCallStatefulPartitionedCall1convTranspose2D1/StatefulPartitionedCall:output:0	bn1_15608	bn1_15610	bn1_15612	bn1_15614*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:��������� *&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *G
fBR@
>__inference_bn1_layer_call_and_return_conditional_losses_14943�
lReLU1/PartitionedCallPartitionedCall$bn1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:��������� * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *J
fERC
A__inference_lReLU1_layer_call_and_return_conditional_losses_15502�
(convTranspose2D2/StatefulPartitionedCallStatefulPartitionedCalllReLU1/PartitionedCall:output:0convtranspose2d2_15618convtranspose2d2_15620*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *T
fORM
K__inference_convTranspose2D2_layer_call_and_return_conditional_losses_15004�
bn2/StatefulPartitionedCallStatefulPartitionedCall1convTranspose2D2/StatefulPartitionedCall:output:0	bn2_15623	bn2_15625	bn2_15627	bn2_15629*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������@*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *G
fBR@
>__inference_bn2_layer_call_and_return_conditional_losses_15051�
lReLU2/PartitionedCallPartitionedCall$bn2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *J
fERC
A__inference_lReLU2_layer_call_and_return_conditional_losses_15523�
(convTranspose2D3/StatefulPartitionedCallStatefulPartitionedCalllReLU2/PartitionedCall:output:0convtranspose2d3_15633convtranspose2d3_15635*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *T
fORM
K__inference_convTranspose2D3_layer_call_and_return_conditional_losses_15112�
bn3/StatefulPartitionedCallStatefulPartitionedCall1convTranspose2D3/StatefulPartitionedCall:output:0	bn3_15638	bn3_15640	bn3_15642	bn3_15644*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������@*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *G
fBR@
>__inference_bn3_layer_call_and_return_conditional_losses_15159�
lReLU3/PartitionedCallPartitionedCall$bn3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *J
fERC
A__inference_lReLU3_layer_call_and_return_conditional_losses_15544�
(convTranspose2D4/StatefulPartitionedCallStatefulPartitionedCalllReLU3/PartitionedCall:output:0convtranspose2d4_15648convtranspose2d4_15650*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *T
fORM
K__inference_convTranspose2D4_layer_call_and_return_conditional_losses_15220�
bn4/StatefulPartitionedCallStatefulPartitionedCall1convTranspose2D4/StatefulPartitionedCall:output:0	bn4_15653	bn4_15655	bn4_15657	bn4_15659*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������@*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *G
fBR@
>__inference_bn4_layer_call_and_return_conditional_losses_15267�
lReLU4/PartitionedCallPartitionedCall$bn4/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *J
fERC
A__inference_lReLU4_layer_call_and_return_conditional_losses_15565�
(convTranspose2D5/StatefulPartitionedCallStatefulPartitionedCalllReLU4/PartitionedCall:output:0convtranspose2d5_15663convtranspose2d5_15665*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:��������� *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *T
fORM
K__inference_convTranspose2D5_layer_call_and_return_conditional_losses_15328�
bn5/StatefulPartitionedCallStatefulPartitionedCall1convTranspose2D5/StatefulPartitionedCall:output:0	bn5_15668	bn5_15670	bn5_15672	bn5_15674*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:��������� *&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *G
fBR@
>__inference_bn5_layer_call_and_return_conditional_losses_15375�
lReLU5/PartitionedCallPartitionedCall$bn5/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:��������� * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *J
fERC
A__inference_lReLU5_layer_call_and_return_conditional_losses_15586�
(convTranspose2D6/StatefulPartitionedCallStatefulPartitionedCalllReLU5/PartitionedCall:output:0convtranspose2d6_15678convtranspose2d6_15680*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *T
fORM
K__inference_convTranspose2D6_layer_call_and_return_conditional_losses_15437�
IdentityIdentity1convTranspose2D6/StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:����������
NoOpNoOp^bn1/StatefulPartitionedCall^bn2/StatefulPartitionedCall^bn3/StatefulPartitionedCall^bn4/StatefulPartitionedCall^bn5/StatefulPartitionedCall)^convTranspose2D1/StatefulPartitionedCall)^convTranspose2D2/StatefulPartitionedCall)^convTranspose2D3/StatefulPartitionedCall)^convTranspose2D4/StatefulPartitionedCall)^convTranspose2D5/StatefulPartitionedCall)^convTranspose2D6/StatefulPartitionedCall^dense1/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*j
_input_shapesY
W:���������
: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2:
bn1/StatefulPartitionedCallbn1/StatefulPartitionedCall2:
bn2/StatefulPartitionedCallbn2/StatefulPartitionedCall2:
bn3/StatefulPartitionedCallbn3/StatefulPartitionedCall2:
bn4/StatefulPartitionedCallbn4/StatefulPartitionedCall2:
bn5/StatefulPartitionedCallbn5/StatefulPartitionedCall2T
(convTranspose2D1/StatefulPartitionedCall(convTranspose2D1/StatefulPartitionedCall2T
(convTranspose2D2/StatefulPartitionedCall(convTranspose2D2/StatefulPartitionedCall2T
(convTranspose2D3/StatefulPartitionedCall(convTranspose2D3/StatefulPartitionedCall2T
(convTranspose2D4/StatefulPartitionedCall(convTranspose2D4/StatefulPartitionedCall2T
(convTranspose2D5/StatefulPartitionedCall(convTranspose2D5/StatefulPartitionedCall2T
(convTranspose2D6/StatefulPartitionedCall(convTranspose2D6/StatefulPartitionedCall2@
dense1/StatefulPartitionedCalldense1/StatefulPartitionedCall:S O
'
_output_shapes
:���������

$
_user_specified_name
inputLayer
�R
�
B__inference_decoder_layer_call_and_return_conditional_losses_15594

inputlayer
dense1_15462:	
�
dense1_15464:	�0
convtranspose2d1_15483:  $
convtranspose2d1_15485: 
	bn1_15488: 
	bn1_15490: 
	bn1_15492: 
	bn1_15494: 0
convtranspose2d2_15504:@ $
convtranspose2d2_15506:@
	bn2_15509:@
	bn2_15511:@
	bn2_15513:@
	bn2_15515:@0
convtranspose2d3_15525:@@$
convtranspose2d3_15527:@
	bn3_15530:@
	bn3_15532:@
	bn3_15534:@
	bn3_15536:@0
convtranspose2d4_15546:@@$
convtranspose2d4_15548:@
	bn4_15551:@
	bn4_15553:@
	bn4_15555:@
	bn4_15557:@0
convtranspose2d5_15567: @$
convtranspose2d5_15569: 
	bn5_15572: 
	bn5_15574: 
	bn5_15576: 
	bn5_15578: 0
convtranspose2d6_15588: $
convtranspose2d6_15590:
identity��bn1/StatefulPartitionedCall�bn2/StatefulPartitionedCall�bn3/StatefulPartitionedCall�bn4/StatefulPartitionedCall�bn5/StatefulPartitionedCall�(convTranspose2D1/StatefulPartitionedCall�(convTranspose2D2/StatefulPartitionedCall�(convTranspose2D3/StatefulPartitionedCall�(convTranspose2D4/StatefulPartitionedCall�(convTranspose2D5/StatefulPartitionedCall�(convTranspose2D6/StatefulPartitionedCall�dense1/StatefulPartitionedCall�
dense1/StatefulPartitionedCallStatefulPartitionedCall
inputlayerdense1_15462dense1_15464*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *J
fERC
A__inference_dense1_layer_call_and_return_conditional_losses_15461�
reshapeLayer/PartitionedCallPartitionedCall'dense1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:��������� * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *P
fKRI
G__inference_reshapeLayer_layer_call_and_return_conditional_losses_15481�
(convTranspose2D1/StatefulPartitionedCallStatefulPartitionedCall%reshapeLayer/PartitionedCall:output:0convtranspose2d1_15483convtranspose2d1_15485*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:��������� *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *T
fORM
K__inference_convTranspose2D1_layer_call_and_return_conditional_losses_14896�
bn1/StatefulPartitionedCallStatefulPartitionedCall1convTranspose2D1/StatefulPartitionedCall:output:0	bn1_15488	bn1_15490	bn1_15492	bn1_15494*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:��������� *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *G
fBR@
>__inference_bn1_layer_call_and_return_conditional_losses_14925�
lReLU1/PartitionedCallPartitionedCall$bn1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:��������� * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *J
fERC
A__inference_lReLU1_layer_call_and_return_conditional_losses_15502�
(convTranspose2D2/StatefulPartitionedCallStatefulPartitionedCalllReLU1/PartitionedCall:output:0convtranspose2d2_15504convtranspose2d2_15506*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *T
fORM
K__inference_convTranspose2D2_layer_call_and_return_conditional_losses_15004�
bn2/StatefulPartitionedCallStatefulPartitionedCall1convTranspose2D2/StatefulPartitionedCall:output:0	bn2_15509	bn2_15511	bn2_15513	bn2_15515*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *G
fBR@
>__inference_bn2_layer_call_and_return_conditional_losses_15033�
lReLU2/PartitionedCallPartitionedCall$bn2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *J
fERC
A__inference_lReLU2_layer_call_and_return_conditional_losses_15523�
(convTranspose2D3/StatefulPartitionedCallStatefulPartitionedCalllReLU2/PartitionedCall:output:0convtranspose2d3_15525convtranspose2d3_15527*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *T
fORM
K__inference_convTranspose2D3_layer_call_and_return_conditional_losses_15112�
bn3/StatefulPartitionedCallStatefulPartitionedCall1convTranspose2D3/StatefulPartitionedCall:output:0	bn3_15530	bn3_15532	bn3_15534	bn3_15536*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *G
fBR@
>__inference_bn3_layer_call_and_return_conditional_losses_15141�
lReLU3/PartitionedCallPartitionedCall$bn3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *J
fERC
A__inference_lReLU3_layer_call_and_return_conditional_losses_15544�
(convTranspose2D4/StatefulPartitionedCallStatefulPartitionedCalllReLU3/PartitionedCall:output:0convtranspose2d4_15546convtranspose2d4_15548*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *T
fORM
K__inference_convTranspose2D4_layer_call_and_return_conditional_losses_15220�
bn4/StatefulPartitionedCallStatefulPartitionedCall1convTranspose2D4/StatefulPartitionedCall:output:0	bn4_15551	bn4_15553	bn4_15555	bn4_15557*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *G
fBR@
>__inference_bn4_layer_call_and_return_conditional_losses_15249�
lReLU4/PartitionedCallPartitionedCall$bn4/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *J
fERC
A__inference_lReLU4_layer_call_and_return_conditional_losses_15565�
(convTranspose2D5/StatefulPartitionedCallStatefulPartitionedCalllReLU4/PartitionedCall:output:0convtranspose2d5_15567convtranspose2d5_15569*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:��������� *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *T
fORM
K__inference_convTranspose2D5_layer_call_and_return_conditional_losses_15328�
bn5/StatefulPartitionedCallStatefulPartitionedCall1convTranspose2D5/StatefulPartitionedCall:output:0	bn5_15572	bn5_15574	bn5_15576	bn5_15578*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:��������� *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *G
fBR@
>__inference_bn5_layer_call_and_return_conditional_losses_15357�
lReLU5/PartitionedCallPartitionedCall$bn5/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:��������� * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *J
fERC
A__inference_lReLU5_layer_call_and_return_conditional_losses_15586�
(convTranspose2D6/StatefulPartitionedCallStatefulPartitionedCalllReLU5/PartitionedCall:output:0convtranspose2d6_15588convtranspose2d6_15590*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *T
fORM
K__inference_convTranspose2D6_layer_call_and_return_conditional_losses_15437�
IdentityIdentity1convTranspose2D6/StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:����������
NoOpNoOp^bn1/StatefulPartitionedCall^bn2/StatefulPartitionedCall^bn3/StatefulPartitionedCall^bn4/StatefulPartitionedCall^bn5/StatefulPartitionedCall)^convTranspose2D1/StatefulPartitionedCall)^convTranspose2D2/StatefulPartitionedCall)^convTranspose2D3/StatefulPartitionedCall)^convTranspose2D4/StatefulPartitionedCall)^convTranspose2D5/StatefulPartitionedCall)^convTranspose2D6/StatefulPartitionedCall^dense1/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*j
_input_shapesY
W:���������
: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2:
bn1/StatefulPartitionedCallbn1/StatefulPartitionedCall2:
bn2/StatefulPartitionedCallbn2/StatefulPartitionedCall2:
bn3/StatefulPartitionedCallbn3/StatefulPartitionedCall2:
bn4/StatefulPartitionedCallbn4/StatefulPartitionedCall2:
bn5/StatefulPartitionedCallbn5/StatefulPartitionedCall2T
(convTranspose2D1/StatefulPartitionedCall(convTranspose2D1/StatefulPartitionedCall2T
(convTranspose2D2/StatefulPartitionedCall(convTranspose2D2/StatefulPartitionedCall2T
(convTranspose2D3/StatefulPartitionedCall(convTranspose2D3/StatefulPartitionedCall2T
(convTranspose2D4/StatefulPartitionedCall(convTranspose2D4/StatefulPartitionedCall2T
(convTranspose2D5/StatefulPartitionedCall(convTranspose2D5/StatefulPartitionedCall2T
(convTranspose2D6/StatefulPartitionedCall(convTranspose2D6/StatefulPartitionedCall2@
dense1/StatefulPartitionedCalldense1/StatefulPartitionedCall:S O
'
_output_shapes
:���������

$
_user_specified_name
inputLayer
� 
�
K__inference_convTranspose2D1_layer_call_and_return_conditional_losses_14896

inputsB
(conv2d_transpose_readvariableop_resource:  -
biasadd_readvariableop_resource: 
identity��BiasAdd/ReadVariableOp�conv2d_transpose/ReadVariableOpI
ShapeShapeinputs*
T0*
_output_shapes
::��]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_1StridedSliceShape:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_2StridedSliceShape:output:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskG
mul/yConst*
_output_shapes
: *
dtype0*
value	B :U
mulMulstrided_slice_1:output:0mul/y:output:0*
T0*
_output_shapes
: I
mul_1/yConst*
_output_shapes
: *
dtype0*
value	B :Y
mul_1Mulstrided_slice_2:output:0mul_1/y:output:0*
T0*
_output_shapes
: I
stack/3Const*
_output_shapes
: *
dtype0*
value	B : y
stackPackstrided_slice:output:0mul:z:0	mul_1:z:0stack/3:output:0*
N*
T0*
_output_shapes
:_
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_3StridedSlicestack:output:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask�
conv2d_transpose/ReadVariableOpReadVariableOp(conv2d_transpose_readvariableop_resource*&
_output_shapes
:  *
dtype0�
conv2d_transposeConv2DBackpropInputstack:output:0'conv2d_transpose/ReadVariableOp:value:0inputs*
T0*A
_output_shapes/
-:+��������������������������� *
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype0�
BiasAddBiasAddconv2d_transpose:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+��������������������������� y
IdentityIdentityBiasAdd:output:0^NoOp*
T0*A
_output_shapes/
-:+��������������������������� �
NoOpNoOp^BiasAdd/ReadVariableOp ^conv2d_transpose/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:+��������������������������� : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2B
conv2d_transpose/ReadVariableOpconv2d_transpose/ReadVariableOp:i e
A
_output_shapes/
-:+��������������������������� 
 
_user_specified_nameinputs
�
�
0__inference_convTranspose2D5_layer_call_fn_17347

inputs!
unknown: @
	unknown_0: 
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+��������������������������� *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *T
fORM
K__inference_convTranspose2D5_layer_call_and_return_conditional_losses_15328�
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+��������������������������� `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:+���������������������������@: : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+���������������������������@
 
_user_specified_nameinputs
�
�
#__inference_bn4_layer_call_fn_17279

inputs
unknown:@
	unknown_0:@
	unknown_1:@
	unknown_2:@
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+���������������������������@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *G
fBR@
>__inference_bn4_layer_call_and_return_conditional_losses_15249�
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+���������������������������@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+���������������������������@: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+���������������������������@
 
_user_specified_nameinputs
�
c
G__inference_reshapeLayer_layer_call_and_return_conditional_losses_16882

inputs
identityI
ShapeShapeinputs*
T0*
_output_shapes
::��]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskQ
Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :Q
Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :Q
Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B : �
Reshape/shapePackstrided_slice:output:0Reshape/shape/1:output:0Reshape/shape/2:output:0Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:l
ReshapeReshapeinputsReshape/shape:output:0*
T0*/
_output_shapes
:��������� `
IdentityIdentityReshape:output:0*
T0*/
_output_shapes
:��������� "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:����������:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
� 
�
K__inference_convTranspose2D4_layer_call_and_return_conditional_losses_17266

inputsB
(conv2d_transpose_readvariableop_resource:@@-
biasadd_readvariableop_resource:@
identity��BiasAdd/ReadVariableOp�conv2d_transpose/ReadVariableOpI
ShapeShapeinputs*
T0*
_output_shapes
::��]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_1StridedSliceShape:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_2StridedSliceShape:output:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskG
mul/yConst*
_output_shapes
: *
dtype0*
value	B :U
mulMulstrided_slice_1:output:0mul/y:output:0*
T0*
_output_shapes
: I
mul_1/yConst*
_output_shapes
: *
dtype0*
value	B :Y
mul_1Mulstrided_slice_2:output:0mul_1/y:output:0*
T0*
_output_shapes
: I
stack/3Const*
_output_shapes
: *
dtype0*
value	B :@y
stackPackstrided_slice:output:0mul:z:0	mul_1:z:0stack/3:output:0*
N*
T0*
_output_shapes
:_
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_3StridedSlicestack:output:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask�
conv2d_transpose/ReadVariableOpReadVariableOp(conv2d_transpose_readvariableop_resource*&
_output_shapes
:@@*
dtype0�
conv2d_transposeConv2DBackpropInputstack:output:0'conv2d_transpose/ReadVariableOp:value:0inputs*
T0*A
_output_shapes/
-:+���������������������������@*
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0�
BiasAddBiasAddconv2d_transpose:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+���������������������������@y
IdentityIdentityBiasAdd:output:0^NoOp*
T0*A
_output_shapes/
-:+���������������������������@�
NoOpNoOp^BiasAdd/ReadVariableOp ^conv2d_transpose/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:+���������������������������@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2B
conv2d_transpose/ReadVariableOpconv2d_transpose/ReadVariableOp:i e
A
_output_shapes/
-:+���������������������������@
 
_user_specified_nameinputs
�
]
A__inference_lReLU2_layer_call_and_return_conditional_losses_15523

inputs
identity_
	LeakyRelu	LeakyReluinputs*/
_output_shapes
:���������@*
alpha%���>g
IdentityIdentityLeakyRelu:activations:0*
T0*/
_output_shapes
:���������@"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������@:W S
/
_output_shapes
:���������@
 
_user_specified_nameinputs
�
�
0__inference_convTranspose2D2_layer_call_fn_17005

inputs!
unknown:@ 
	unknown_0:@
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+���������������������������@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *T
fORM
K__inference_convTranspose2D2_layer_call_and_return_conditional_losses_15004�
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+���������������������������@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:+��������������������������� : : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+��������������������������� 
 
_user_specified_nameinputs
�
c
G__inference_reshapeLayer_layer_call_and_return_conditional_losses_15481

inputs
identityI
ShapeShapeinputs*
T0*
_output_shapes
::��]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskQ
Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :Q
Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :Q
Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B : �
Reshape/shapePackstrided_slice:output:0Reshape/shape/1:output:0Reshape/shape/2:output:0Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:l
ReshapeReshapeinputsReshape/shape:output:0*
T0*/
_output_shapes
:��������� `
IdentityIdentityReshape:output:0*
T0*/
_output_shapes
:��������� "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:����������:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
]
A__inference_lReLU1_layer_call_and_return_conditional_losses_16996

inputs
identity_
	LeakyRelu	LeakyReluinputs*/
_output_shapes
:��������� *
alpha%���>g
IdentityIdentityLeakyRelu:activations:0*
T0*/
_output_shapes
:��������� "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:��������� :W S
/
_output_shapes
:��������� 
 
_user_specified_nameinputs
�
�
>__inference_bn5_layer_call_and_return_conditional_losses_15375

inputs%
readvariableop_resource: '
readvariableop_1_resource: 6
(fusedbatchnormv3_readvariableop_resource: 8
*fusedbatchnormv3_readvariableop_1_resource: 
identity��FusedBatchNormV3/ReadVariableOp�!FusedBatchNormV3/ReadVariableOp_1�ReadVariableOp�ReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype0�
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype0�
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype0�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+��������������������������� : : : : :*
epsilon%o�:*
is_training( }
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+��������������������������� �
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+��������������������������� : : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+��������������������������� 
 
_user_specified_nameinputs
� 
�
K__inference_convTranspose2D3_layer_call_and_return_conditional_losses_15112

inputsB
(conv2d_transpose_readvariableop_resource:@@-
biasadd_readvariableop_resource:@
identity��BiasAdd/ReadVariableOp�conv2d_transpose/ReadVariableOpI
ShapeShapeinputs*
T0*
_output_shapes
::��]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_1StridedSliceShape:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_2StridedSliceShape:output:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskG
mul/yConst*
_output_shapes
: *
dtype0*
value	B :U
mulMulstrided_slice_1:output:0mul/y:output:0*
T0*
_output_shapes
: I
mul_1/yConst*
_output_shapes
: *
dtype0*
value	B :Y
mul_1Mulstrided_slice_2:output:0mul_1/y:output:0*
T0*
_output_shapes
: I
stack/3Const*
_output_shapes
: *
dtype0*
value	B :@y
stackPackstrided_slice:output:0mul:z:0	mul_1:z:0stack/3:output:0*
N*
T0*
_output_shapes
:_
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_3StridedSlicestack:output:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask�
conv2d_transpose/ReadVariableOpReadVariableOp(conv2d_transpose_readvariableop_resource*&
_output_shapes
:@@*
dtype0�
conv2d_transposeConv2DBackpropInputstack:output:0'conv2d_transpose/ReadVariableOp:value:0inputs*
T0*A
_output_shapes/
-:+���������������������������@*
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0�
BiasAddBiasAddconv2d_transpose:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+���������������������������@y
IdentityIdentityBiasAdd:output:0^NoOp*
T0*A
_output_shapes/
-:+���������������������������@�
NoOpNoOp^BiasAdd/ReadVariableOp ^conv2d_transpose/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:+���������������������������@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2B
conv2d_transpose/ReadVariableOpconv2d_transpose/ReadVariableOp:i e
A
_output_shapes/
-:+���������������������������@
 
_user_specified_nameinputs
�R
�
B__inference_decoder_layer_call_and_return_conditional_losses_15940

inputs
dense1_15853:	
�
dense1_15855:	�0
convtranspose2d1_15859:  $
convtranspose2d1_15861: 
	bn1_15864: 
	bn1_15866: 
	bn1_15868: 
	bn1_15870: 0
convtranspose2d2_15874:@ $
convtranspose2d2_15876:@
	bn2_15879:@
	bn2_15881:@
	bn2_15883:@
	bn2_15885:@0
convtranspose2d3_15889:@@$
convtranspose2d3_15891:@
	bn3_15894:@
	bn3_15896:@
	bn3_15898:@
	bn3_15900:@0
convtranspose2d4_15904:@@$
convtranspose2d4_15906:@
	bn4_15909:@
	bn4_15911:@
	bn4_15913:@
	bn4_15915:@0
convtranspose2d5_15919: @$
convtranspose2d5_15921: 
	bn5_15924: 
	bn5_15926: 
	bn5_15928: 
	bn5_15930: 0
convtranspose2d6_15934: $
convtranspose2d6_15936:
identity��bn1/StatefulPartitionedCall�bn2/StatefulPartitionedCall�bn3/StatefulPartitionedCall�bn4/StatefulPartitionedCall�bn5/StatefulPartitionedCall�(convTranspose2D1/StatefulPartitionedCall�(convTranspose2D2/StatefulPartitionedCall�(convTranspose2D3/StatefulPartitionedCall�(convTranspose2D4/StatefulPartitionedCall�(convTranspose2D5/StatefulPartitionedCall�(convTranspose2D6/StatefulPartitionedCall�dense1/StatefulPartitionedCall�
dense1/StatefulPartitionedCallStatefulPartitionedCallinputsdense1_15853dense1_15855*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *J
fERC
A__inference_dense1_layer_call_and_return_conditional_losses_15461�
reshapeLayer/PartitionedCallPartitionedCall'dense1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:��������� * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *P
fKRI
G__inference_reshapeLayer_layer_call_and_return_conditional_losses_15481�
(convTranspose2D1/StatefulPartitionedCallStatefulPartitionedCall%reshapeLayer/PartitionedCall:output:0convtranspose2d1_15859convtranspose2d1_15861*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:��������� *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *T
fORM
K__inference_convTranspose2D1_layer_call_and_return_conditional_losses_14896�
bn1/StatefulPartitionedCallStatefulPartitionedCall1convTranspose2D1/StatefulPartitionedCall:output:0	bn1_15864	bn1_15866	bn1_15868	bn1_15870*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:��������� *&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *G
fBR@
>__inference_bn1_layer_call_and_return_conditional_losses_14943�
lReLU1/PartitionedCallPartitionedCall$bn1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:��������� * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *J
fERC
A__inference_lReLU1_layer_call_and_return_conditional_losses_15502�
(convTranspose2D2/StatefulPartitionedCallStatefulPartitionedCalllReLU1/PartitionedCall:output:0convtranspose2d2_15874convtranspose2d2_15876*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *T
fORM
K__inference_convTranspose2D2_layer_call_and_return_conditional_losses_15004�
bn2/StatefulPartitionedCallStatefulPartitionedCall1convTranspose2D2/StatefulPartitionedCall:output:0	bn2_15879	bn2_15881	bn2_15883	bn2_15885*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������@*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *G
fBR@
>__inference_bn2_layer_call_and_return_conditional_losses_15051�
lReLU2/PartitionedCallPartitionedCall$bn2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *J
fERC
A__inference_lReLU2_layer_call_and_return_conditional_losses_15523�
(convTranspose2D3/StatefulPartitionedCallStatefulPartitionedCalllReLU2/PartitionedCall:output:0convtranspose2d3_15889convtranspose2d3_15891*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *T
fORM
K__inference_convTranspose2D3_layer_call_and_return_conditional_losses_15112�
bn3/StatefulPartitionedCallStatefulPartitionedCall1convTranspose2D3/StatefulPartitionedCall:output:0	bn3_15894	bn3_15896	bn3_15898	bn3_15900*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������@*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *G
fBR@
>__inference_bn3_layer_call_and_return_conditional_losses_15159�
lReLU3/PartitionedCallPartitionedCall$bn3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *J
fERC
A__inference_lReLU3_layer_call_and_return_conditional_losses_15544�
(convTranspose2D4/StatefulPartitionedCallStatefulPartitionedCalllReLU3/PartitionedCall:output:0convtranspose2d4_15904convtranspose2d4_15906*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *T
fORM
K__inference_convTranspose2D4_layer_call_and_return_conditional_losses_15220�
bn4/StatefulPartitionedCallStatefulPartitionedCall1convTranspose2D4/StatefulPartitionedCall:output:0	bn4_15909	bn4_15911	bn4_15913	bn4_15915*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������@*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *G
fBR@
>__inference_bn4_layer_call_and_return_conditional_losses_15267�
lReLU4/PartitionedCallPartitionedCall$bn4/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *J
fERC
A__inference_lReLU4_layer_call_and_return_conditional_losses_15565�
(convTranspose2D5/StatefulPartitionedCallStatefulPartitionedCalllReLU4/PartitionedCall:output:0convtranspose2d5_15919convtranspose2d5_15921*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:��������� *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *T
fORM
K__inference_convTranspose2D5_layer_call_and_return_conditional_losses_15328�
bn5/StatefulPartitionedCallStatefulPartitionedCall1convTranspose2D5/StatefulPartitionedCall:output:0	bn5_15924	bn5_15926	bn5_15928	bn5_15930*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:��������� *&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *G
fBR@
>__inference_bn5_layer_call_and_return_conditional_losses_15375�
lReLU5/PartitionedCallPartitionedCall$bn5/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:��������� * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *J
fERC
A__inference_lReLU5_layer_call_and_return_conditional_losses_15586�
(convTranspose2D6/StatefulPartitionedCallStatefulPartitionedCalllReLU5/PartitionedCall:output:0convtranspose2d6_15934convtranspose2d6_15936*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *T
fORM
K__inference_convTranspose2D6_layer_call_and_return_conditional_losses_15437�
IdentityIdentity1convTranspose2D6/StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:����������
NoOpNoOp^bn1/StatefulPartitionedCall^bn2/StatefulPartitionedCall^bn3/StatefulPartitionedCall^bn4/StatefulPartitionedCall^bn5/StatefulPartitionedCall)^convTranspose2D1/StatefulPartitionedCall)^convTranspose2D2/StatefulPartitionedCall)^convTranspose2D3/StatefulPartitionedCall)^convTranspose2D4/StatefulPartitionedCall)^convTranspose2D5/StatefulPartitionedCall)^convTranspose2D6/StatefulPartitionedCall^dense1/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*j
_input_shapesY
W:���������
: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2:
bn1/StatefulPartitionedCallbn1/StatefulPartitionedCall2:
bn2/StatefulPartitionedCallbn2/StatefulPartitionedCall2:
bn3/StatefulPartitionedCallbn3/StatefulPartitionedCall2:
bn4/StatefulPartitionedCallbn4/StatefulPartitionedCall2:
bn5/StatefulPartitionedCallbn5/StatefulPartitionedCall2T
(convTranspose2D1/StatefulPartitionedCall(convTranspose2D1/StatefulPartitionedCall2T
(convTranspose2D2/StatefulPartitionedCall(convTranspose2D2/StatefulPartitionedCall2T
(convTranspose2D3/StatefulPartitionedCall(convTranspose2D3/StatefulPartitionedCall2T
(convTranspose2D4/StatefulPartitionedCall(convTranspose2D4/StatefulPartitionedCall2T
(convTranspose2D5/StatefulPartitionedCall(convTranspose2D5/StatefulPartitionedCall2T
(convTranspose2D6/StatefulPartitionedCall(convTranspose2D6/StatefulPartitionedCall2@
dense1/StatefulPartitionedCalldense1/StatefulPartitionedCall:O K
'
_output_shapes
:���������

 
_user_specified_nameinputs
�
�
#__inference_bn4_layer_call_fn_17292

inputs
unknown:@
	unknown_0:@
	unknown_1:@
	unknown_2:@
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+���������������������������@*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *G
fBR@
>__inference_bn4_layer_call_and_return_conditional_losses_15267�
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+���������������������������@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+���������������������������@: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+���������������������������@
 
_user_specified_nameinputs
�
�
>__inference_bn1_layer_call_and_return_conditional_losses_16968

inputs%
readvariableop_resource: '
readvariableop_1_resource: 6
(fusedbatchnormv3_readvariableop_resource: 8
*fusedbatchnormv3_readvariableop_1_resource: 
identity��AssignNewValue�AssignNewValue_1�FusedBatchNormV3/ReadVariableOp�!FusedBatchNormV3/ReadVariableOp_1�ReadVariableOp�ReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype0�
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype0�
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype0�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+��������������������������� : : : : :*
epsilon%o�:*
exponential_avg_factor%
�#<�
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(�
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(}
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+��������������������������� �
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+��������������������������� : : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+��������������������������� 
 
_user_specified_nameinputs
�R
�
B__inference_decoder_layer_call_and_return_conditional_losses_15777

inputs
dense1_15690:	
�
dense1_15692:	�0
convtranspose2d1_15696:  $
convtranspose2d1_15698: 
	bn1_15701: 
	bn1_15703: 
	bn1_15705: 
	bn1_15707: 0
convtranspose2d2_15711:@ $
convtranspose2d2_15713:@
	bn2_15716:@
	bn2_15718:@
	bn2_15720:@
	bn2_15722:@0
convtranspose2d3_15726:@@$
convtranspose2d3_15728:@
	bn3_15731:@
	bn3_15733:@
	bn3_15735:@
	bn3_15737:@0
convtranspose2d4_15741:@@$
convtranspose2d4_15743:@
	bn4_15746:@
	bn4_15748:@
	bn4_15750:@
	bn4_15752:@0
convtranspose2d5_15756: @$
convtranspose2d5_15758: 
	bn5_15761: 
	bn5_15763: 
	bn5_15765: 
	bn5_15767: 0
convtranspose2d6_15771: $
convtranspose2d6_15773:
identity��bn1/StatefulPartitionedCall�bn2/StatefulPartitionedCall�bn3/StatefulPartitionedCall�bn4/StatefulPartitionedCall�bn5/StatefulPartitionedCall�(convTranspose2D1/StatefulPartitionedCall�(convTranspose2D2/StatefulPartitionedCall�(convTranspose2D3/StatefulPartitionedCall�(convTranspose2D4/StatefulPartitionedCall�(convTranspose2D5/StatefulPartitionedCall�(convTranspose2D6/StatefulPartitionedCall�dense1/StatefulPartitionedCall�
dense1/StatefulPartitionedCallStatefulPartitionedCallinputsdense1_15690dense1_15692*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *J
fERC
A__inference_dense1_layer_call_and_return_conditional_losses_15461�
reshapeLayer/PartitionedCallPartitionedCall'dense1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:��������� * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *P
fKRI
G__inference_reshapeLayer_layer_call_and_return_conditional_losses_15481�
(convTranspose2D1/StatefulPartitionedCallStatefulPartitionedCall%reshapeLayer/PartitionedCall:output:0convtranspose2d1_15696convtranspose2d1_15698*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:��������� *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *T
fORM
K__inference_convTranspose2D1_layer_call_and_return_conditional_losses_14896�
bn1/StatefulPartitionedCallStatefulPartitionedCall1convTranspose2D1/StatefulPartitionedCall:output:0	bn1_15701	bn1_15703	bn1_15705	bn1_15707*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:��������� *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *G
fBR@
>__inference_bn1_layer_call_and_return_conditional_losses_14925�
lReLU1/PartitionedCallPartitionedCall$bn1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:��������� * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *J
fERC
A__inference_lReLU1_layer_call_and_return_conditional_losses_15502�
(convTranspose2D2/StatefulPartitionedCallStatefulPartitionedCalllReLU1/PartitionedCall:output:0convtranspose2d2_15711convtranspose2d2_15713*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *T
fORM
K__inference_convTranspose2D2_layer_call_and_return_conditional_losses_15004�
bn2/StatefulPartitionedCallStatefulPartitionedCall1convTranspose2D2/StatefulPartitionedCall:output:0	bn2_15716	bn2_15718	bn2_15720	bn2_15722*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *G
fBR@
>__inference_bn2_layer_call_and_return_conditional_losses_15033�
lReLU2/PartitionedCallPartitionedCall$bn2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *J
fERC
A__inference_lReLU2_layer_call_and_return_conditional_losses_15523�
(convTranspose2D3/StatefulPartitionedCallStatefulPartitionedCalllReLU2/PartitionedCall:output:0convtranspose2d3_15726convtranspose2d3_15728*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *T
fORM
K__inference_convTranspose2D3_layer_call_and_return_conditional_losses_15112�
bn3/StatefulPartitionedCallStatefulPartitionedCall1convTranspose2D3/StatefulPartitionedCall:output:0	bn3_15731	bn3_15733	bn3_15735	bn3_15737*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *G
fBR@
>__inference_bn3_layer_call_and_return_conditional_losses_15141�
lReLU3/PartitionedCallPartitionedCall$bn3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *J
fERC
A__inference_lReLU3_layer_call_and_return_conditional_losses_15544�
(convTranspose2D4/StatefulPartitionedCallStatefulPartitionedCalllReLU3/PartitionedCall:output:0convtranspose2d4_15741convtranspose2d4_15743*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *T
fORM
K__inference_convTranspose2D4_layer_call_and_return_conditional_losses_15220�
bn4/StatefulPartitionedCallStatefulPartitionedCall1convTranspose2D4/StatefulPartitionedCall:output:0	bn4_15746	bn4_15748	bn4_15750	bn4_15752*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *G
fBR@
>__inference_bn4_layer_call_and_return_conditional_losses_15249�
lReLU4/PartitionedCallPartitionedCall$bn4/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *J
fERC
A__inference_lReLU4_layer_call_and_return_conditional_losses_15565�
(convTranspose2D5/StatefulPartitionedCallStatefulPartitionedCalllReLU4/PartitionedCall:output:0convtranspose2d5_15756convtranspose2d5_15758*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:��������� *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *T
fORM
K__inference_convTranspose2D5_layer_call_and_return_conditional_losses_15328�
bn5/StatefulPartitionedCallStatefulPartitionedCall1convTranspose2D5/StatefulPartitionedCall:output:0	bn5_15761	bn5_15763	bn5_15765	bn5_15767*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:��������� *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *G
fBR@
>__inference_bn5_layer_call_and_return_conditional_losses_15357�
lReLU5/PartitionedCallPartitionedCall$bn5/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:��������� * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *J
fERC
A__inference_lReLU5_layer_call_and_return_conditional_losses_15586�
(convTranspose2D6/StatefulPartitionedCallStatefulPartitionedCalllReLU5/PartitionedCall:output:0convtranspose2d6_15771convtranspose2d6_15773*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *T
fORM
K__inference_convTranspose2D6_layer_call_and_return_conditional_losses_15437�
IdentityIdentity1convTranspose2D6/StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:����������
NoOpNoOp^bn1/StatefulPartitionedCall^bn2/StatefulPartitionedCall^bn3/StatefulPartitionedCall^bn4/StatefulPartitionedCall^bn5/StatefulPartitionedCall)^convTranspose2D1/StatefulPartitionedCall)^convTranspose2D2/StatefulPartitionedCall)^convTranspose2D3/StatefulPartitionedCall)^convTranspose2D4/StatefulPartitionedCall)^convTranspose2D5/StatefulPartitionedCall)^convTranspose2D6/StatefulPartitionedCall^dense1/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*j
_input_shapesY
W:���������
: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2:
bn1/StatefulPartitionedCallbn1/StatefulPartitionedCall2:
bn2/StatefulPartitionedCallbn2/StatefulPartitionedCall2:
bn3/StatefulPartitionedCallbn3/StatefulPartitionedCall2:
bn4/StatefulPartitionedCallbn4/StatefulPartitionedCall2:
bn5/StatefulPartitionedCallbn5/StatefulPartitionedCall2T
(convTranspose2D1/StatefulPartitionedCall(convTranspose2D1/StatefulPartitionedCall2T
(convTranspose2D2/StatefulPartitionedCall(convTranspose2D2/StatefulPartitionedCall2T
(convTranspose2D3/StatefulPartitionedCall(convTranspose2D3/StatefulPartitionedCall2T
(convTranspose2D4/StatefulPartitionedCall(convTranspose2D4/StatefulPartitionedCall2T
(convTranspose2D5/StatefulPartitionedCall(convTranspose2D5/StatefulPartitionedCall2T
(convTranspose2D6/StatefulPartitionedCall(convTranspose2D6/StatefulPartitionedCall2@
dense1/StatefulPartitionedCalldense1/StatefulPartitionedCall:O K
'
_output_shapes
:���������

 
_user_specified_nameinputs
�
H
,__inference_reshapeLayer_layer_call_fn_16868

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:��������� * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *P
fKRI
G__inference_reshapeLayer_layer_call_and_return_conditional_losses_15481h
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:��������� "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:����������:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�	
�
A__inference_dense1_layer_call_and_return_conditional_losses_16863

inputs1
matmul_readvariableop_resource:	
�.
biasadd_readvariableop_resource:	�
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	
�*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������`
IdentityIdentityBiasAdd:output:0^NoOp*
T0*(
_output_shapes
:����������w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������
: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:���������

 
_user_specified_nameinputs
�
]
A__inference_lReLU3_layer_call_and_return_conditional_losses_17224

inputs
identity_
	LeakyRelu	LeakyReluinputs*/
_output_shapes
:���������@*
alpha%���>g
IdentityIdentityLeakyRelu:activations:0*
T0*/
_output_shapes
:���������@"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������@:W S
/
_output_shapes
:���������@
 
_user_specified_nameinputs
�
B
&__inference_lReLU5_layer_call_fn_17447

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:��������� * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *J
fERC
A__inference_lReLU5_layer_call_and_return_conditional_losses_15586h
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:��������� "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:��������� :W S
/
_output_shapes
:��������� 
 
_user_specified_nameinputs
� 
�
K__inference_convTranspose2D2_layer_call_and_return_conditional_losses_15004

inputsB
(conv2d_transpose_readvariableop_resource:@ -
biasadd_readvariableop_resource:@
identity��BiasAdd/ReadVariableOp�conv2d_transpose/ReadVariableOpI
ShapeShapeinputs*
T0*
_output_shapes
::��]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_1StridedSliceShape:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_2StridedSliceShape:output:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskG
mul/yConst*
_output_shapes
: *
dtype0*
value	B :U
mulMulstrided_slice_1:output:0mul/y:output:0*
T0*
_output_shapes
: I
mul_1/yConst*
_output_shapes
: *
dtype0*
value	B :Y
mul_1Mulstrided_slice_2:output:0mul_1/y:output:0*
T0*
_output_shapes
: I
stack/3Const*
_output_shapes
: *
dtype0*
value	B :@y
stackPackstrided_slice:output:0mul:z:0	mul_1:z:0stack/3:output:0*
N*
T0*
_output_shapes
:_
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_3StridedSlicestack:output:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask�
conv2d_transpose/ReadVariableOpReadVariableOp(conv2d_transpose_readvariableop_resource*&
_output_shapes
:@ *
dtype0�
conv2d_transposeConv2DBackpropInputstack:output:0'conv2d_transpose/ReadVariableOp:value:0inputs*
T0*A
_output_shapes/
-:+���������������������������@*
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0�
BiasAddBiasAddconv2d_transpose:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+���������������������������@y
IdentityIdentityBiasAdd:output:0^NoOp*
T0*A
_output_shapes/
-:+���������������������������@�
NoOpNoOp^BiasAdd/ReadVariableOp ^conv2d_transpose/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:+��������������������������� : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2B
conv2d_transpose/ReadVariableOpconv2d_transpose/ReadVariableOp:i e
A
_output_shapes/
-:+��������������������������� 
 
_user_specified_nameinputs
�
]
A__inference_lReLU5_layer_call_and_return_conditional_losses_15586

inputs
identity_
	LeakyRelu	LeakyReluinputs*/
_output_shapes
:��������� *
alpha%���>g
IdentityIdentityLeakyRelu:activations:0*
T0*/
_output_shapes
:��������� "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:��������� :W S
/
_output_shapes
:��������� 
 
_user_specified_nameinputs
�
]
A__inference_lReLU2_layer_call_and_return_conditional_losses_17110

inputs
identity_
	LeakyRelu	LeakyReluinputs*/
_output_shapes
:���������@*
alpha%���>g
IdentityIdentityLeakyRelu:activations:0*
T0*/
_output_shapes
:���������@"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������@:W S
/
_output_shapes
:���������@
 
_user_specified_nameinputs
�
�
'__inference_decoder_layer_call_fn_16011

inputlayer
unknown:	
�
	unknown_0:	�#
	unknown_1:  
	unknown_2: 
	unknown_3: 
	unknown_4: 
	unknown_5: 
	unknown_6: #
	unknown_7:@ 
	unknown_8:@
	unknown_9:@

unknown_10:@

unknown_11:@

unknown_12:@$

unknown_13:@@

unknown_14:@

unknown_15:@

unknown_16:@

unknown_17:@

unknown_18:@$

unknown_19:@@

unknown_20:@

unknown_21:@

unknown_22:@

unknown_23:@

unknown_24:@$

unknown_25: @

unknown_26: 

unknown_27: 

unknown_28: 

unknown_29: 

unknown_30: $

unknown_31: 

unknown_32:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCall
inputlayerunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28
unknown_29
unknown_30
unknown_31
unknown_32*.
Tin'
%2#*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������*D
_read_only_resource_inputs&
$"	
 !"*-
config_proto

CPU

GPU 2J 8� *K
fFRD
B__inference_decoder_layer_call_and_return_conditional_losses_15940w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:���������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*j
_input_shapesY
W:���������
: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
'
_output_shapes
:���������

$
_user_specified_name
inputLayer
�
�
#__inference_bn5_layer_call_fn_17393

inputs
unknown: 
	unknown_0: 
	unknown_1: 
	unknown_2: 
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+��������������������������� *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *G
fBR@
>__inference_bn5_layer_call_and_return_conditional_losses_15357�
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+��������������������������� `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+��������������������������� : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+��������������������������� 
 
_user_specified_nameinputs
�
�
#__inference_bn3_layer_call_fn_17165

inputs
unknown:@
	unknown_0:@
	unknown_1:@
	unknown_2:@
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+���������������������������@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *G
fBR@
>__inference_bn3_layer_call_and_return_conditional_losses_15141�
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+���������������������������@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+���������������������������@: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+���������������������������@
 
_user_specified_nameinputs
�
�
'__inference_decoder_layer_call_fn_15848

inputlayer
unknown:	
�
	unknown_0:	�#
	unknown_1:  
	unknown_2: 
	unknown_3: 
	unknown_4: 
	unknown_5: 
	unknown_6: #
	unknown_7:@ 
	unknown_8:@
	unknown_9:@

unknown_10:@

unknown_11:@

unknown_12:@$

unknown_13:@@

unknown_14:@

unknown_15:@

unknown_16:@

unknown_17:@

unknown_18:@$

unknown_19:@@

unknown_20:@

unknown_21:@

unknown_22:@

unknown_23:@

unknown_24:@$

unknown_25: @

unknown_26: 

unknown_27: 

unknown_28: 

unknown_29: 

unknown_30: $

unknown_31: 

unknown_32:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCall
inputlayerunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28
unknown_29
unknown_30
unknown_31
unknown_32*.
Tin'
%2#*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������*:
_read_only_resource_inputs
	
!"*-
config_proto

CPU

GPU 2J 8� *K
fFRD
B__inference_decoder_layer_call_and_return_conditional_losses_15777w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:���������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*j
_input_shapesY
W:���������
: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
'
_output_shapes
:���������

$
_user_specified_name
inputLayer
�
B
&__inference_lReLU1_layer_call_fn_16991

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:��������� * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *J
fERC
A__inference_lReLU1_layer_call_and_return_conditional_losses_15502h
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:��������� "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:��������� :W S
/
_output_shapes
:��������� 
 
_user_specified_nameinputs
�
�
>__inference_bn4_layer_call_and_return_conditional_losses_17328

inputs%
readvariableop_resource:@'
readvariableop_1_resource:@6
(fusedbatchnormv3_readvariableop_resource:@8
*fusedbatchnormv3_readvariableop_1_resource:@
identity��FusedBatchNormV3/ReadVariableOp�!FusedBatchNormV3/ReadVariableOp_1�ReadVariableOp�ReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype0�
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype0�
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype0�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+���������������������������@:@:@:@:@:*
epsilon%o�:*
is_training( }
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+���������������������������@�
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+���������������������������@: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+���������������������������@
 
_user_specified_nameinputs
�
�
>__inference_bn3_layer_call_and_return_conditional_losses_15159

inputs%
readvariableop_resource:@'
readvariableop_1_resource:@6
(fusedbatchnormv3_readvariableop_resource:@8
*fusedbatchnormv3_readvariableop_1_resource:@
identity��FusedBatchNormV3/ReadVariableOp�!FusedBatchNormV3/ReadVariableOp_1�ReadVariableOp�ReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype0�
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype0�
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype0�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+���������������������������@:@:@:@:@:*
epsilon%o�:*
is_training( }
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+���������������������������@�
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+���������������������������@: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+���������������������������@
 
_user_specified_nameinputs
�
�
>__inference_bn4_layer_call_and_return_conditional_losses_15249

inputs%
readvariableop_resource:@'
readvariableop_1_resource:@6
(fusedbatchnormv3_readvariableop_resource:@8
*fusedbatchnormv3_readvariableop_1_resource:@
identity��AssignNewValue�AssignNewValue_1�FusedBatchNormV3/ReadVariableOp�!FusedBatchNormV3/ReadVariableOp_1�ReadVariableOp�ReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype0�
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype0�
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype0�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+���������������������������@:@:@:@:@:*
epsilon%o�:*
exponential_avg_factor%
�#<�
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(�
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(}
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+���������������������������@�
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+���������������������������@: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+���������������������������@
 
_user_specified_nameinputs
�
�
>__inference_bn1_layer_call_and_return_conditional_losses_16986

inputs%
readvariableop_resource: '
readvariableop_1_resource: 6
(fusedbatchnormv3_readvariableop_resource: 8
*fusedbatchnormv3_readvariableop_1_resource: 
identity��FusedBatchNormV3/ReadVariableOp�!FusedBatchNormV3/ReadVariableOp_1�ReadVariableOp�ReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype0�
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype0�
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype0�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+��������������������������� : : : : :*
epsilon%o�:*
is_training( }
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+��������������������������� �
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+��������������������������� : : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+��������������������������� 
 
_user_specified_nameinputs
�
�
>__inference_bn2_layer_call_and_return_conditional_losses_17082

inputs%
readvariableop_resource:@'
readvariableop_1_resource:@6
(fusedbatchnormv3_readvariableop_resource:@8
*fusedbatchnormv3_readvariableop_1_resource:@
identity��AssignNewValue�AssignNewValue_1�FusedBatchNormV3/ReadVariableOp�!FusedBatchNormV3/ReadVariableOp_1�ReadVariableOp�ReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype0�
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype0�
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype0�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+���������������������������@:@:@:@:@:*
epsilon%o�:*
exponential_avg_factor%
�#<�
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(�
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(}
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+���������������������������@�
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+���������������������������@: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+���������������������������@
 
_user_specified_nameinputs
�
�
#__inference_bn2_layer_call_fn_17051

inputs
unknown:@
	unknown_0:@
	unknown_1:@
	unknown_2:@
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+���������������������������@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *G
fBR@
>__inference_bn2_layer_call_and_return_conditional_losses_15033�
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+���������������������������@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+���������������������������@: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+���������������������������@
 
_user_specified_nameinputs
�
�
0__inference_convTranspose2D1_layer_call_fn_16891

inputs!
unknown:  
	unknown_0: 
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+��������������������������� *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *T
fORM
K__inference_convTranspose2D1_layer_call_and_return_conditional_losses_14896�
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+��������������������������� `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:+��������������������������� : : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+��������������������������� 
 
_user_specified_nameinputs
ؐ
�
!__inference__traced_restore_17834
file_prefix1
assignvariableop_dense1_kernel:	
�-
assignvariableop_1_dense1_bias:	�D
*assignvariableop_2_convtranspose2d1_kernel:  6
(assignvariableop_3_convtranspose2d1_bias: *
assignvariableop_4_bn1_gamma: )
assignvariableop_5_bn1_beta: 0
"assignvariableop_6_bn1_moving_mean: 4
&assignvariableop_7_bn1_moving_variance: D
*assignvariableop_8_convtranspose2d2_kernel:@ 6
(assignvariableop_9_convtranspose2d2_bias:@+
assignvariableop_10_bn2_gamma:@*
assignvariableop_11_bn2_beta:@1
#assignvariableop_12_bn2_moving_mean:@5
'assignvariableop_13_bn2_moving_variance:@E
+assignvariableop_14_convtranspose2d3_kernel:@@7
)assignvariableop_15_convtranspose2d3_bias:@+
assignvariableop_16_bn3_gamma:@*
assignvariableop_17_bn3_beta:@1
#assignvariableop_18_bn3_moving_mean:@5
'assignvariableop_19_bn3_moving_variance:@E
+assignvariableop_20_convtranspose2d4_kernel:@@7
)assignvariableop_21_convtranspose2d4_bias:@+
assignvariableop_22_bn4_gamma:@*
assignvariableop_23_bn4_beta:@1
#assignvariableop_24_bn4_moving_mean:@5
'assignvariableop_25_bn4_moving_variance:@E
+assignvariableop_26_convtranspose2d5_kernel: @7
)assignvariableop_27_convtranspose2d5_bias: +
assignvariableop_28_bn5_gamma: *
assignvariableop_29_bn5_beta: 1
#assignvariableop_30_bn5_moving_mean: 5
'assignvariableop_31_bn5_moving_variance: E
+assignvariableop_32_convtranspose2d6_kernel: 7
)assignvariableop_33_convtranspose2d6_bias:
identity_35��AssignVariableOp�AssignVariableOp_1�AssignVariableOp_10�AssignVariableOp_11�AssignVariableOp_12�AssignVariableOp_13�AssignVariableOp_14�AssignVariableOp_15�AssignVariableOp_16�AssignVariableOp_17�AssignVariableOp_18�AssignVariableOp_19�AssignVariableOp_2�AssignVariableOp_20�AssignVariableOp_21�AssignVariableOp_22�AssignVariableOp_23�AssignVariableOp_24�AssignVariableOp_25�AssignVariableOp_26�AssignVariableOp_27�AssignVariableOp_28�AssignVariableOp_29�AssignVariableOp_3�AssignVariableOp_30�AssignVariableOp_31�AssignVariableOp_32�AssignVariableOp_33�AssignVariableOp_4�AssignVariableOp_5�AssignVariableOp_6�AssignVariableOp_7�AssignVariableOp_8�AssignVariableOp_9�
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:#*
dtype0*�
value�B�#B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-2/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-2/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-2/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-4/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-4/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-4/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-6/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-6/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-6/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-7/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-7/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-8/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-8/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-8/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-8/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-9/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-9/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-10/gamma/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-10/beta/.ATTRIBUTES/VARIABLE_VALUEB<layer_with_weights-10/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-10/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-11/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-11/bias/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH�
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:#*
dtype0*Y
valuePBN#B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B �
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*�
_output_shapes�
�:::::::::::::::::::::::::::::::::::*1
dtypes'
%2#[
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOpAssignVariableOpassignvariableop_dense1_kernelIdentity:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_1AssignVariableOpassignvariableop_1_dense1_biasIdentity_1:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_2AssignVariableOp*assignvariableop_2_convtranspose2d1_kernelIdentity_2:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_3AssignVariableOp(assignvariableop_3_convtranspose2d1_biasIdentity_3:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_4AssignVariableOpassignvariableop_4_bn1_gammaIdentity_4:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_5AssignVariableOpassignvariableop_5_bn1_betaIdentity_5:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_6AssignVariableOp"assignvariableop_6_bn1_moving_meanIdentity_6:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_7AssignVariableOp&assignvariableop_7_bn1_moving_varianceIdentity_7:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_8AssignVariableOp*assignvariableop_8_convtranspose2d2_kernelIdentity_8:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_9AssignVariableOp(assignvariableop_9_convtranspose2d2_biasIdentity_9:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_10AssignVariableOpassignvariableop_10_bn2_gammaIdentity_10:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_11AssignVariableOpassignvariableop_11_bn2_betaIdentity_11:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_12AssignVariableOp#assignvariableop_12_bn2_moving_meanIdentity_12:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_13AssignVariableOp'assignvariableop_13_bn2_moving_varianceIdentity_13:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_14AssignVariableOp+assignvariableop_14_convtranspose2d3_kernelIdentity_14:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_15AssignVariableOp)assignvariableop_15_convtranspose2d3_biasIdentity_15:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_16AssignVariableOpassignvariableop_16_bn3_gammaIdentity_16:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_17AssignVariableOpassignvariableop_17_bn3_betaIdentity_17:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_18AssignVariableOp#assignvariableop_18_bn3_moving_meanIdentity_18:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_19AssignVariableOp'assignvariableop_19_bn3_moving_varianceIdentity_19:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_20AssignVariableOp+assignvariableop_20_convtranspose2d4_kernelIdentity_20:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_21AssignVariableOp)assignvariableop_21_convtranspose2d4_biasIdentity_21:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_22AssignVariableOpassignvariableop_22_bn4_gammaIdentity_22:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_23AssignVariableOpassignvariableop_23_bn4_betaIdentity_23:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_24AssignVariableOp#assignvariableop_24_bn4_moving_meanIdentity_24:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_25AssignVariableOp'assignvariableop_25_bn4_moving_varianceIdentity_25:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_26AssignVariableOp+assignvariableop_26_convtranspose2d5_kernelIdentity_26:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_27AssignVariableOp)assignvariableop_27_convtranspose2d5_biasIdentity_27:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_28AssignVariableOpassignvariableop_28_bn5_gammaIdentity_28:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_29AssignVariableOpassignvariableop_29_bn5_betaIdentity_29:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_30AssignVariableOp#assignvariableop_30_bn5_moving_meanIdentity_30:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_31AssignVariableOp'assignvariableop_31_bn5_moving_varianceIdentity_31:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_32AssignVariableOp+assignvariableop_32_convtranspose2d6_kernelIdentity_32:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_33IdentityRestoreV2:tensors:33"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_33AssignVariableOp)assignvariableop_33_convtranspose2d6_biasIdentity_33:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0Y
NoOpNoOp"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 �
Identity_34Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: W
Identity_35IdentityIdentity_34:output:0^NoOp_1*
T0*
_output_shapes
: �
NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*"
_acd_function_control_output(*
_output_shapes
 "#
identity_35Identity_35:output:0*Y
_input_shapesH
F: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132*
AssignVariableOp_14AssignVariableOp_142*
AssignVariableOp_15AssignVariableOp_152*
AssignVariableOp_16AssignVariableOp_162*
AssignVariableOp_17AssignVariableOp_172*
AssignVariableOp_18AssignVariableOp_182*
AssignVariableOp_19AssignVariableOp_192(
AssignVariableOp_2AssignVariableOp_22*
AssignVariableOp_20AssignVariableOp_202*
AssignVariableOp_21AssignVariableOp_212*
AssignVariableOp_22AssignVariableOp_222*
AssignVariableOp_23AssignVariableOp_232*
AssignVariableOp_24AssignVariableOp_242*
AssignVariableOp_25AssignVariableOp_252*
AssignVariableOp_26AssignVariableOp_262*
AssignVariableOp_27AssignVariableOp_272*
AssignVariableOp_28AssignVariableOp_282*
AssignVariableOp_29AssignVariableOp_292(
AssignVariableOp_3AssignVariableOp_32*
AssignVariableOp_30AssignVariableOp_302*
AssignVariableOp_31AssignVariableOp_312*
AssignVariableOp_32AssignVariableOp_322*
AssignVariableOp_33AssignVariableOp_332(
AssignVariableOp_4AssignVariableOp_42(
AssignVariableOp_5AssignVariableOp_52(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_9:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
�
�
>__inference_bn5_layer_call_and_return_conditional_losses_17442

inputs%
readvariableop_resource: '
readvariableop_1_resource: 6
(fusedbatchnormv3_readvariableop_resource: 8
*fusedbatchnormv3_readvariableop_1_resource: 
identity��FusedBatchNormV3/ReadVariableOp�!FusedBatchNormV3/ReadVariableOp_1�ReadVariableOp�ReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype0�
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype0�
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype0�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+��������������������������� : : : : :*
epsilon%o�:*
is_training( }
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+��������������������������� �
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+��������������������������� : : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+��������������������������� 
 
_user_specified_nameinputs
��
�
__inference__traced_save_17722
file_prefix7
$read_disablecopyonread_dense1_kernel:	
�3
$read_1_disablecopyonread_dense1_bias:	�J
0read_2_disablecopyonread_convtranspose2d1_kernel:  <
.read_3_disablecopyonread_convtranspose2d1_bias: 0
"read_4_disablecopyonread_bn1_gamma: /
!read_5_disablecopyonread_bn1_beta: 6
(read_6_disablecopyonread_bn1_moving_mean: :
,read_7_disablecopyonread_bn1_moving_variance: J
0read_8_disablecopyonread_convtranspose2d2_kernel:@ <
.read_9_disablecopyonread_convtranspose2d2_bias:@1
#read_10_disablecopyonread_bn2_gamma:@0
"read_11_disablecopyonread_bn2_beta:@7
)read_12_disablecopyonread_bn2_moving_mean:@;
-read_13_disablecopyonread_bn2_moving_variance:@K
1read_14_disablecopyonread_convtranspose2d3_kernel:@@=
/read_15_disablecopyonread_convtranspose2d3_bias:@1
#read_16_disablecopyonread_bn3_gamma:@0
"read_17_disablecopyonread_bn3_beta:@7
)read_18_disablecopyonread_bn3_moving_mean:@;
-read_19_disablecopyonread_bn3_moving_variance:@K
1read_20_disablecopyonread_convtranspose2d4_kernel:@@=
/read_21_disablecopyonread_convtranspose2d4_bias:@1
#read_22_disablecopyonread_bn4_gamma:@0
"read_23_disablecopyonread_bn4_beta:@7
)read_24_disablecopyonread_bn4_moving_mean:@;
-read_25_disablecopyonread_bn4_moving_variance:@K
1read_26_disablecopyonread_convtranspose2d5_kernel: @=
/read_27_disablecopyonread_convtranspose2d5_bias: 1
#read_28_disablecopyonread_bn5_gamma: 0
"read_29_disablecopyonread_bn5_beta: 7
)read_30_disablecopyonread_bn5_moving_mean: ;
-read_31_disablecopyonread_bn5_moving_variance: K
1read_32_disablecopyonread_convtranspose2d6_kernel: =
/read_33_disablecopyonread_convtranspose2d6_bias:
savev2_const
identity_69��MergeV2Checkpoints�Read/DisableCopyOnRead�Read/ReadVariableOp�Read_1/DisableCopyOnRead�Read_1/ReadVariableOp�Read_10/DisableCopyOnRead�Read_10/ReadVariableOp�Read_11/DisableCopyOnRead�Read_11/ReadVariableOp�Read_12/DisableCopyOnRead�Read_12/ReadVariableOp�Read_13/DisableCopyOnRead�Read_13/ReadVariableOp�Read_14/DisableCopyOnRead�Read_14/ReadVariableOp�Read_15/DisableCopyOnRead�Read_15/ReadVariableOp�Read_16/DisableCopyOnRead�Read_16/ReadVariableOp�Read_17/DisableCopyOnRead�Read_17/ReadVariableOp�Read_18/DisableCopyOnRead�Read_18/ReadVariableOp�Read_19/DisableCopyOnRead�Read_19/ReadVariableOp�Read_2/DisableCopyOnRead�Read_2/ReadVariableOp�Read_20/DisableCopyOnRead�Read_20/ReadVariableOp�Read_21/DisableCopyOnRead�Read_21/ReadVariableOp�Read_22/DisableCopyOnRead�Read_22/ReadVariableOp�Read_23/DisableCopyOnRead�Read_23/ReadVariableOp�Read_24/DisableCopyOnRead�Read_24/ReadVariableOp�Read_25/DisableCopyOnRead�Read_25/ReadVariableOp�Read_26/DisableCopyOnRead�Read_26/ReadVariableOp�Read_27/DisableCopyOnRead�Read_27/ReadVariableOp�Read_28/DisableCopyOnRead�Read_28/ReadVariableOp�Read_29/DisableCopyOnRead�Read_29/ReadVariableOp�Read_3/DisableCopyOnRead�Read_3/ReadVariableOp�Read_30/DisableCopyOnRead�Read_30/ReadVariableOp�Read_31/DisableCopyOnRead�Read_31/ReadVariableOp�Read_32/DisableCopyOnRead�Read_32/ReadVariableOp�Read_33/DisableCopyOnRead�Read_33/ReadVariableOp�Read_4/DisableCopyOnRead�Read_4/ReadVariableOp�Read_5/DisableCopyOnRead�Read_5/ReadVariableOp�Read_6/DisableCopyOnRead�Read_6/ReadVariableOp�Read_7/DisableCopyOnRead�Read_7/ReadVariableOp�Read_8/DisableCopyOnRead�Read_8/ReadVariableOp�Read_9/DisableCopyOnRead�Read_9/ReadVariableOpw
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*Z
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.parta
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/part�
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: f

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: L

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :f
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : �
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: v
Read/DisableCopyOnReadDisableCopyOnRead$read_disablecopyonread_dense1_kernel"/device:CPU:0*
_output_shapes
 �
Read/ReadVariableOpReadVariableOp$read_disablecopyonread_dense1_kernel^Read/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:	
�*
dtype0j
IdentityIdentityRead/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:	
�b

Identity_1IdentityIdentity:output:0"/device:CPU:0*
T0*
_output_shapes
:	
�x
Read_1/DisableCopyOnReadDisableCopyOnRead$read_1_disablecopyonread_dense1_bias"/device:CPU:0*
_output_shapes
 �
Read_1/ReadVariableOpReadVariableOp$read_1_disablecopyonread_dense1_bias^Read_1/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:�*
dtype0j

Identity_2IdentityRead_1/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:�`

Identity_3IdentityIdentity_2:output:0"/device:CPU:0*
T0*
_output_shapes	
:��
Read_2/DisableCopyOnReadDisableCopyOnRead0read_2_disablecopyonread_convtranspose2d1_kernel"/device:CPU:0*
_output_shapes
 �
Read_2/ReadVariableOpReadVariableOp0read_2_disablecopyonread_convtranspose2d1_kernel^Read_2/DisableCopyOnRead"/device:CPU:0*&
_output_shapes
:  *
dtype0u

Identity_4IdentityRead_2/ReadVariableOp:value:0"/device:CPU:0*
T0*&
_output_shapes
:  k

Identity_5IdentityIdentity_4:output:0"/device:CPU:0*
T0*&
_output_shapes
:  �
Read_3/DisableCopyOnReadDisableCopyOnRead.read_3_disablecopyonread_convtranspose2d1_bias"/device:CPU:0*
_output_shapes
 �
Read_3/ReadVariableOpReadVariableOp.read_3_disablecopyonread_convtranspose2d1_bias^Read_3/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0i

Identity_6IdentityRead_3/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: _

Identity_7IdentityIdentity_6:output:0"/device:CPU:0*
T0*
_output_shapes
: v
Read_4/DisableCopyOnReadDisableCopyOnRead"read_4_disablecopyonread_bn1_gamma"/device:CPU:0*
_output_shapes
 �
Read_4/ReadVariableOpReadVariableOp"read_4_disablecopyonread_bn1_gamma^Read_4/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0i

Identity_8IdentityRead_4/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: _

Identity_9IdentityIdentity_8:output:0"/device:CPU:0*
T0*
_output_shapes
: u
Read_5/DisableCopyOnReadDisableCopyOnRead!read_5_disablecopyonread_bn1_beta"/device:CPU:0*
_output_shapes
 �
Read_5/ReadVariableOpReadVariableOp!read_5_disablecopyonread_bn1_beta^Read_5/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0j
Identity_10IdentityRead_5/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: a
Identity_11IdentityIdentity_10:output:0"/device:CPU:0*
T0*
_output_shapes
: |
Read_6/DisableCopyOnReadDisableCopyOnRead(read_6_disablecopyonread_bn1_moving_mean"/device:CPU:0*
_output_shapes
 �
Read_6/ReadVariableOpReadVariableOp(read_6_disablecopyonread_bn1_moving_mean^Read_6/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0j
Identity_12IdentityRead_6/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: a
Identity_13IdentityIdentity_12:output:0"/device:CPU:0*
T0*
_output_shapes
: �
Read_7/DisableCopyOnReadDisableCopyOnRead,read_7_disablecopyonread_bn1_moving_variance"/device:CPU:0*
_output_shapes
 �
Read_7/ReadVariableOpReadVariableOp,read_7_disablecopyonread_bn1_moving_variance^Read_7/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0j
Identity_14IdentityRead_7/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: a
Identity_15IdentityIdentity_14:output:0"/device:CPU:0*
T0*
_output_shapes
: �
Read_8/DisableCopyOnReadDisableCopyOnRead0read_8_disablecopyonread_convtranspose2d2_kernel"/device:CPU:0*
_output_shapes
 �
Read_8/ReadVariableOpReadVariableOp0read_8_disablecopyonread_convtranspose2d2_kernel^Read_8/DisableCopyOnRead"/device:CPU:0*&
_output_shapes
:@ *
dtype0v
Identity_16IdentityRead_8/ReadVariableOp:value:0"/device:CPU:0*
T0*&
_output_shapes
:@ m
Identity_17IdentityIdentity_16:output:0"/device:CPU:0*
T0*&
_output_shapes
:@ �
Read_9/DisableCopyOnReadDisableCopyOnRead.read_9_disablecopyonread_convtranspose2d2_bias"/device:CPU:0*
_output_shapes
 �
Read_9/ReadVariableOpReadVariableOp.read_9_disablecopyonread_convtranspose2d2_bias^Read_9/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:@*
dtype0j
Identity_18IdentityRead_9/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:@a
Identity_19IdentityIdentity_18:output:0"/device:CPU:0*
T0*
_output_shapes
:@x
Read_10/DisableCopyOnReadDisableCopyOnRead#read_10_disablecopyonread_bn2_gamma"/device:CPU:0*
_output_shapes
 �
Read_10/ReadVariableOpReadVariableOp#read_10_disablecopyonread_bn2_gamma^Read_10/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:@*
dtype0k
Identity_20IdentityRead_10/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:@a
Identity_21IdentityIdentity_20:output:0"/device:CPU:0*
T0*
_output_shapes
:@w
Read_11/DisableCopyOnReadDisableCopyOnRead"read_11_disablecopyonread_bn2_beta"/device:CPU:0*
_output_shapes
 �
Read_11/ReadVariableOpReadVariableOp"read_11_disablecopyonread_bn2_beta^Read_11/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:@*
dtype0k
Identity_22IdentityRead_11/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:@a
Identity_23IdentityIdentity_22:output:0"/device:CPU:0*
T0*
_output_shapes
:@~
Read_12/DisableCopyOnReadDisableCopyOnRead)read_12_disablecopyonread_bn2_moving_mean"/device:CPU:0*
_output_shapes
 �
Read_12/ReadVariableOpReadVariableOp)read_12_disablecopyonread_bn2_moving_mean^Read_12/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:@*
dtype0k
Identity_24IdentityRead_12/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:@a
Identity_25IdentityIdentity_24:output:0"/device:CPU:0*
T0*
_output_shapes
:@�
Read_13/DisableCopyOnReadDisableCopyOnRead-read_13_disablecopyonread_bn2_moving_variance"/device:CPU:0*
_output_shapes
 �
Read_13/ReadVariableOpReadVariableOp-read_13_disablecopyonread_bn2_moving_variance^Read_13/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:@*
dtype0k
Identity_26IdentityRead_13/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:@a
Identity_27IdentityIdentity_26:output:0"/device:CPU:0*
T0*
_output_shapes
:@�
Read_14/DisableCopyOnReadDisableCopyOnRead1read_14_disablecopyonread_convtranspose2d3_kernel"/device:CPU:0*
_output_shapes
 �
Read_14/ReadVariableOpReadVariableOp1read_14_disablecopyonread_convtranspose2d3_kernel^Read_14/DisableCopyOnRead"/device:CPU:0*&
_output_shapes
:@@*
dtype0w
Identity_28IdentityRead_14/ReadVariableOp:value:0"/device:CPU:0*
T0*&
_output_shapes
:@@m
Identity_29IdentityIdentity_28:output:0"/device:CPU:0*
T0*&
_output_shapes
:@@�
Read_15/DisableCopyOnReadDisableCopyOnRead/read_15_disablecopyonread_convtranspose2d3_bias"/device:CPU:0*
_output_shapes
 �
Read_15/ReadVariableOpReadVariableOp/read_15_disablecopyonread_convtranspose2d3_bias^Read_15/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:@*
dtype0k
Identity_30IdentityRead_15/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:@a
Identity_31IdentityIdentity_30:output:0"/device:CPU:0*
T0*
_output_shapes
:@x
Read_16/DisableCopyOnReadDisableCopyOnRead#read_16_disablecopyonread_bn3_gamma"/device:CPU:0*
_output_shapes
 �
Read_16/ReadVariableOpReadVariableOp#read_16_disablecopyonread_bn3_gamma^Read_16/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:@*
dtype0k
Identity_32IdentityRead_16/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:@a
Identity_33IdentityIdentity_32:output:0"/device:CPU:0*
T0*
_output_shapes
:@w
Read_17/DisableCopyOnReadDisableCopyOnRead"read_17_disablecopyonread_bn3_beta"/device:CPU:0*
_output_shapes
 �
Read_17/ReadVariableOpReadVariableOp"read_17_disablecopyonread_bn3_beta^Read_17/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:@*
dtype0k
Identity_34IdentityRead_17/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:@a
Identity_35IdentityIdentity_34:output:0"/device:CPU:0*
T0*
_output_shapes
:@~
Read_18/DisableCopyOnReadDisableCopyOnRead)read_18_disablecopyonread_bn3_moving_mean"/device:CPU:0*
_output_shapes
 �
Read_18/ReadVariableOpReadVariableOp)read_18_disablecopyonread_bn3_moving_mean^Read_18/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:@*
dtype0k
Identity_36IdentityRead_18/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:@a
Identity_37IdentityIdentity_36:output:0"/device:CPU:0*
T0*
_output_shapes
:@�
Read_19/DisableCopyOnReadDisableCopyOnRead-read_19_disablecopyonread_bn3_moving_variance"/device:CPU:0*
_output_shapes
 �
Read_19/ReadVariableOpReadVariableOp-read_19_disablecopyonread_bn3_moving_variance^Read_19/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:@*
dtype0k
Identity_38IdentityRead_19/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:@a
Identity_39IdentityIdentity_38:output:0"/device:CPU:0*
T0*
_output_shapes
:@�
Read_20/DisableCopyOnReadDisableCopyOnRead1read_20_disablecopyonread_convtranspose2d4_kernel"/device:CPU:0*
_output_shapes
 �
Read_20/ReadVariableOpReadVariableOp1read_20_disablecopyonread_convtranspose2d4_kernel^Read_20/DisableCopyOnRead"/device:CPU:0*&
_output_shapes
:@@*
dtype0w
Identity_40IdentityRead_20/ReadVariableOp:value:0"/device:CPU:0*
T0*&
_output_shapes
:@@m
Identity_41IdentityIdentity_40:output:0"/device:CPU:0*
T0*&
_output_shapes
:@@�
Read_21/DisableCopyOnReadDisableCopyOnRead/read_21_disablecopyonread_convtranspose2d4_bias"/device:CPU:0*
_output_shapes
 �
Read_21/ReadVariableOpReadVariableOp/read_21_disablecopyonread_convtranspose2d4_bias^Read_21/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:@*
dtype0k
Identity_42IdentityRead_21/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:@a
Identity_43IdentityIdentity_42:output:0"/device:CPU:0*
T0*
_output_shapes
:@x
Read_22/DisableCopyOnReadDisableCopyOnRead#read_22_disablecopyonread_bn4_gamma"/device:CPU:0*
_output_shapes
 �
Read_22/ReadVariableOpReadVariableOp#read_22_disablecopyonread_bn4_gamma^Read_22/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:@*
dtype0k
Identity_44IdentityRead_22/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:@a
Identity_45IdentityIdentity_44:output:0"/device:CPU:0*
T0*
_output_shapes
:@w
Read_23/DisableCopyOnReadDisableCopyOnRead"read_23_disablecopyonread_bn4_beta"/device:CPU:0*
_output_shapes
 �
Read_23/ReadVariableOpReadVariableOp"read_23_disablecopyonread_bn4_beta^Read_23/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:@*
dtype0k
Identity_46IdentityRead_23/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:@a
Identity_47IdentityIdentity_46:output:0"/device:CPU:0*
T0*
_output_shapes
:@~
Read_24/DisableCopyOnReadDisableCopyOnRead)read_24_disablecopyonread_bn4_moving_mean"/device:CPU:0*
_output_shapes
 �
Read_24/ReadVariableOpReadVariableOp)read_24_disablecopyonread_bn4_moving_mean^Read_24/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:@*
dtype0k
Identity_48IdentityRead_24/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:@a
Identity_49IdentityIdentity_48:output:0"/device:CPU:0*
T0*
_output_shapes
:@�
Read_25/DisableCopyOnReadDisableCopyOnRead-read_25_disablecopyonread_bn4_moving_variance"/device:CPU:0*
_output_shapes
 �
Read_25/ReadVariableOpReadVariableOp-read_25_disablecopyonread_bn4_moving_variance^Read_25/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:@*
dtype0k
Identity_50IdentityRead_25/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:@a
Identity_51IdentityIdentity_50:output:0"/device:CPU:0*
T0*
_output_shapes
:@�
Read_26/DisableCopyOnReadDisableCopyOnRead1read_26_disablecopyonread_convtranspose2d5_kernel"/device:CPU:0*
_output_shapes
 �
Read_26/ReadVariableOpReadVariableOp1read_26_disablecopyonread_convtranspose2d5_kernel^Read_26/DisableCopyOnRead"/device:CPU:0*&
_output_shapes
: @*
dtype0w
Identity_52IdentityRead_26/ReadVariableOp:value:0"/device:CPU:0*
T0*&
_output_shapes
: @m
Identity_53IdentityIdentity_52:output:0"/device:CPU:0*
T0*&
_output_shapes
: @�
Read_27/DisableCopyOnReadDisableCopyOnRead/read_27_disablecopyonread_convtranspose2d5_bias"/device:CPU:0*
_output_shapes
 �
Read_27/ReadVariableOpReadVariableOp/read_27_disablecopyonread_convtranspose2d5_bias^Read_27/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0k
Identity_54IdentityRead_27/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: a
Identity_55IdentityIdentity_54:output:0"/device:CPU:0*
T0*
_output_shapes
: x
Read_28/DisableCopyOnReadDisableCopyOnRead#read_28_disablecopyonread_bn5_gamma"/device:CPU:0*
_output_shapes
 �
Read_28/ReadVariableOpReadVariableOp#read_28_disablecopyonread_bn5_gamma^Read_28/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0k
Identity_56IdentityRead_28/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: a
Identity_57IdentityIdentity_56:output:0"/device:CPU:0*
T0*
_output_shapes
: w
Read_29/DisableCopyOnReadDisableCopyOnRead"read_29_disablecopyonread_bn5_beta"/device:CPU:0*
_output_shapes
 �
Read_29/ReadVariableOpReadVariableOp"read_29_disablecopyonread_bn5_beta^Read_29/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0k
Identity_58IdentityRead_29/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: a
Identity_59IdentityIdentity_58:output:0"/device:CPU:0*
T0*
_output_shapes
: ~
Read_30/DisableCopyOnReadDisableCopyOnRead)read_30_disablecopyonread_bn5_moving_mean"/device:CPU:0*
_output_shapes
 �
Read_30/ReadVariableOpReadVariableOp)read_30_disablecopyonread_bn5_moving_mean^Read_30/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0k
Identity_60IdentityRead_30/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: a
Identity_61IdentityIdentity_60:output:0"/device:CPU:0*
T0*
_output_shapes
: �
Read_31/DisableCopyOnReadDisableCopyOnRead-read_31_disablecopyonread_bn5_moving_variance"/device:CPU:0*
_output_shapes
 �
Read_31/ReadVariableOpReadVariableOp-read_31_disablecopyonread_bn5_moving_variance^Read_31/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0k
Identity_62IdentityRead_31/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: a
Identity_63IdentityIdentity_62:output:0"/device:CPU:0*
T0*
_output_shapes
: �
Read_32/DisableCopyOnReadDisableCopyOnRead1read_32_disablecopyonread_convtranspose2d6_kernel"/device:CPU:0*
_output_shapes
 �
Read_32/ReadVariableOpReadVariableOp1read_32_disablecopyonread_convtranspose2d6_kernel^Read_32/DisableCopyOnRead"/device:CPU:0*&
_output_shapes
: *
dtype0w
Identity_64IdentityRead_32/ReadVariableOp:value:0"/device:CPU:0*
T0*&
_output_shapes
: m
Identity_65IdentityIdentity_64:output:0"/device:CPU:0*
T0*&
_output_shapes
: �
Read_33/DisableCopyOnReadDisableCopyOnRead/read_33_disablecopyonread_convtranspose2d6_bias"/device:CPU:0*
_output_shapes
 �
Read_33/ReadVariableOpReadVariableOp/read_33_disablecopyonread_convtranspose2d6_bias^Read_33/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0k
Identity_66IdentityRead_33/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:a
Identity_67IdentityIdentity_66:output:0"/device:CPU:0*
T0*
_output_shapes
:�
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:#*
dtype0*�
value�B�#B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-2/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-2/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-2/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-4/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-4/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-4/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-6/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-6/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-6/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-7/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-7/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-8/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-8/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-8/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-8/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-9/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-9/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-10/gamma/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-10/beta/.ATTRIBUTES/VARIABLE_VALUEB<layer_with_weights-10/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-10/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-11/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-11/bias/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH�
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:#*
dtype0*Y
valuePBN#B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B �
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0Identity_1:output:0Identity_3:output:0Identity_5:output:0Identity_7:output:0Identity_9:output:0Identity_11:output:0Identity_13:output:0Identity_15:output:0Identity_17:output:0Identity_19:output:0Identity_21:output:0Identity_23:output:0Identity_25:output:0Identity_27:output:0Identity_29:output:0Identity_31:output:0Identity_33:output:0Identity_35:output:0Identity_37:output:0Identity_39:output:0Identity_41:output:0Identity_43:output:0Identity_45:output:0Identity_47:output:0Identity_49:output:0Identity_51:output:0Identity_53:output:0Identity_55:output:0Identity_57:output:0Identity_59:output:0Identity_61:output:0Identity_63:output:0Identity_65:output:0Identity_67:output:0savev2_const"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *1
dtypes'
%2#�
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:�
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 i
Identity_68Identityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: U
Identity_69IdentityIdentity_68:output:0^NoOp*
T0*
_output_shapes
: �
NoOpNoOp^MergeV2Checkpoints^Read/DisableCopyOnRead^Read/ReadVariableOp^Read_1/DisableCopyOnRead^Read_1/ReadVariableOp^Read_10/DisableCopyOnRead^Read_10/ReadVariableOp^Read_11/DisableCopyOnRead^Read_11/ReadVariableOp^Read_12/DisableCopyOnRead^Read_12/ReadVariableOp^Read_13/DisableCopyOnRead^Read_13/ReadVariableOp^Read_14/DisableCopyOnRead^Read_14/ReadVariableOp^Read_15/DisableCopyOnRead^Read_15/ReadVariableOp^Read_16/DisableCopyOnRead^Read_16/ReadVariableOp^Read_17/DisableCopyOnRead^Read_17/ReadVariableOp^Read_18/DisableCopyOnRead^Read_18/ReadVariableOp^Read_19/DisableCopyOnRead^Read_19/ReadVariableOp^Read_2/DisableCopyOnRead^Read_2/ReadVariableOp^Read_20/DisableCopyOnRead^Read_20/ReadVariableOp^Read_21/DisableCopyOnRead^Read_21/ReadVariableOp^Read_22/DisableCopyOnRead^Read_22/ReadVariableOp^Read_23/DisableCopyOnRead^Read_23/ReadVariableOp^Read_24/DisableCopyOnRead^Read_24/ReadVariableOp^Read_25/DisableCopyOnRead^Read_25/ReadVariableOp^Read_26/DisableCopyOnRead^Read_26/ReadVariableOp^Read_27/DisableCopyOnRead^Read_27/ReadVariableOp^Read_28/DisableCopyOnRead^Read_28/ReadVariableOp^Read_29/DisableCopyOnRead^Read_29/ReadVariableOp^Read_3/DisableCopyOnRead^Read_3/ReadVariableOp^Read_30/DisableCopyOnRead^Read_30/ReadVariableOp^Read_31/DisableCopyOnRead^Read_31/ReadVariableOp^Read_32/DisableCopyOnRead^Read_32/ReadVariableOp^Read_33/DisableCopyOnRead^Read_33/ReadVariableOp^Read_4/DisableCopyOnRead^Read_4/ReadVariableOp^Read_5/DisableCopyOnRead^Read_5/ReadVariableOp^Read_6/DisableCopyOnRead^Read_6/ReadVariableOp^Read_7/DisableCopyOnRead^Read_7/ReadVariableOp^Read_8/DisableCopyOnRead^Read_8/ReadVariableOp^Read_9/DisableCopyOnRead^Read_9/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "#
identity_69Identity_69:output:0*[
_input_shapesJ
H: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2(
MergeV2CheckpointsMergeV2Checkpoints20
Read/DisableCopyOnReadRead/DisableCopyOnRead2*
Read/ReadVariableOpRead/ReadVariableOp24
Read_1/DisableCopyOnReadRead_1/DisableCopyOnRead2.
Read_1/ReadVariableOpRead_1/ReadVariableOp26
Read_10/DisableCopyOnReadRead_10/DisableCopyOnRead20
Read_10/ReadVariableOpRead_10/ReadVariableOp26
Read_11/DisableCopyOnReadRead_11/DisableCopyOnRead20
Read_11/ReadVariableOpRead_11/ReadVariableOp26
Read_12/DisableCopyOnReadRead_12/DisableCopyOnRead20
Read_12/ReadVariableOpRead_12/ReadVariableOp26
Read_13/DisableCopyOnReadRead_13/DisableCopyOnRead20
Read_13/ReadVariableOpRead_13/ReadVariableOp26
Read_14/DisableCopyOnReadRead_14/DisableCopyOnRead20
Read_14/ReadVariableOpRead_14/ReadVariableOp26
Read_15/DisableCopyOnReadRead_15/DisableCopyOnRead20
Read_15/ReadVariableOpRead_15/ReadVariableOp26
Read_16/DisableCopyOnReadRead_16/DisableCopyOnRead20
Read_16/ReadVariableOpRead_16/ReadVariableOp26
Read_17/DisableCopyOnReadRead_17/DisableCopyOnRead20
Read_17/ReadVariableOpRead_17/ReadVariableOp26
Read_18/DisableCopyOnReadRead_18/DisableCopyOnRead20
Read_18/ReadVariableOpRead_18/ReadVariableOp26
Read_19/DisableCopyOnReadRead_19/DisableCopyOnRead20
Read_19/ReadVariableOpRead_19/ReadVariableOp24
Read_2/DisableCopyOnReadRead_2/DisableCopyOnRead2.
Read_2/ReadVariableOpRead_2/ReadVariableOp26
Read_20/DisableCopyOnReadRead_20/DisableCopyOnRead20
Read_20/ReadVariableOpRead_20/ReadVariableOp26
Read_21/DisableCopyOnReadRead_21/DisableCopyOnRead20
Read_21/ReadVariableOpRead_21/ReadVariableOp26
Read_22/DisableCopyOnReadRead_22/DisableCopyOnRead20
Read_22/ReadVariableOpRead_22/ReadVariableOp26
Read_23/DisableCopyOnReadRead_23/DisableCopyOnRead20
Read_23/ReadVariableOpRead_23/ReadVariableOp26
Read_24/DisableCopyOnReadRead_24/DisableCopyOnRead20
Read_24/ReadVariableOpRead_24/ReadVariableOp26
Read_25/DisableCopyOnReadRead_25/DisableCopyOnRead20
Read_25/ReadVariableOpRead_25/ReadVariableOp26
Read_26/DisableCopyOnReadRead_26/DisableCopyOnRead20
Read_26/ReadVariableOpRead_26/ReadVariableOp26
Read_27/DisableCopyOnReadRead_27/DisableCopyOnRead20
Read_27/ReadVariableOpRead_27/ReadVariableOp26
Read_28/DisableCopyOnReadRead_28/DisableCopyOnRead20
Read_28/ReadVariableOpRead_28/ReadVariableOp26
Read_29/DisableCopyOnReadRead_29/DisableCopyOnRead20
Read_29/ReadVariableOpRead_29/ReadVariableOp24
Read_3/DisableCopyOnReadRead_3/DisableCopyOnRead2.
Read_3/ReadVariableOpRead_3/ReadVariableOp26
Read_30/DisableCopyOnReadRead_30/DisableCopyOnRead20
Read_30/ReadVariableOpRead_30/ReadVariableOp26
Read_31/DisableCopyOnReadRead_31/DisableCopyOnRead20
Read_31/ReadVariableOpRead_31/ReadVariableOp26
Read_32/DisableCopyOnReadRead_32/DisableCopyOnRead20
Read_32/ReadVariableOpRead_32/ReadVariableOp26
Read_33/DisableCopyOnReadRead_33/DisableCopyOnRead20
Read_33/ReadVariableOpRead_33/ReadVariableOp24
Read_4/DisableCopyOnReadRead_4/DisableCopyOnRead2.
Read_4/ReadVariableOpRead_4/ReadVariableOp24
Read_5/DisableCopyOnReadRead_5/DisableCopyOnRead2.
Read_5/ReadVariableOpRead_5/ReadVariableOp24
Read_6/DisableCopyOnReadRead_6/DisableCopyOnRead2.
Read_6/ReadVariableOpRead_6/ReadVariableOp24
Read_7/DisableCopyOnReadRead_7/DisableCopyOnRead2.
Read_7/ReadVariableOpRead_7/ReadVariableOp24
Read_8/DisableCopyOnReadRead_8/DisableCopyOnRead2.
Read_8/ReadVariableOpRead_8/ReadVariableOp24
Read_9/DisableCopyOnReadRead_9/DisableCopyOnRead2.
Read_9/ReadVariableOpRead_9/ReadVariableOp:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:#

_output_shapes
: 
�
]
A__inference_lReLU1_layer_call_and_return_conditional_losses_15502

inputs
identity_
	LeakyRelu	LeakyReluinputs*/
_output_shapes
:��������� *
alpha%���>g
IdentityIdentityLeakyRelu:activations:0*
T0*/
_output_shapes
:��������� "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:��������� :W S
/
_output_shapes
:��������� 
 
_user_specified_nameinputs
�
�
0__inference_convTranspose2D6_layer_call_fn_17461

inputs!
unknown: 
	unknown_0:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+���������������������������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *T
fORM
K__inference_convTranspose2D6_layer_call_and_return_conditional_losses_15437�
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+���������������������������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:+��������������������������� : : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+��������������������������� 
 
_user_specified_nameinputs
�
�
>__inference_bn5_layer_call_and_return_conditional_losses_15357

inputs%
readvariableop_resource: '
readvariableop_1_resource: 6
(fusedbatchnormv3_readvariableop_resource: 8
*fusedbatchnormv3_readvariableop_1_resource: 
identity��AssignNewValue�AssignNewValue_1�FusedBatchNormV3/ReadVariableOp�!FusedBatchNormV3/ReadVariableOp_1�ReadVariableOp�ReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype0�
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype0�
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype0�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+��������������������������� : : : : :*
epsilon%o�:*
exponential_avg_factor%
�#<�
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(�
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(}
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+��������������������������� �
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+��������������������������� : : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+��������������������������� 
 
_user_specified_nameinputs
� 
�
K__inference_convTranspose2D5_layer_call_and_return_conditional_losses_17380

inputsB
(conv2d_transpose_readvariableop_resource: @-
biasadd_readvariableop_resource: 
identity��BiasAdd/ReadVariableOp�conv2d_transpose/ReadVariableOpI
ShapeShapeinputs*
T0*
_output_shapes
::��]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_1StridedSliceShape:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_2StridedSliceShape:output:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskG
mul/yConst*
_output_shapes
: *
dtype0*
value	B :U
mulMulstrided_slice_1:output:0mul/y:output:0*
T0*
_output_shapes
: I
mul_1/yConst*
_output_shapes
: *
dtype0*
value	B :Y
mul_1Mulstrided_slice_2:output:0mul_1/y:output:0*
T0*
_output_shapes
: I
stack/3Const*
_output_shapes
: *
dtype0*
value	B : y
stackPackstrided_slice:output:0mul:z:0	mul_1:z:0stack/3:output:0*
N*
T0*
_output_shapes
:_
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_3StridedSlicestack:output:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask�
conv2d_transpose/ReadVariableOpReadVariableOp(conv2d_transpose_readvariableop_resource*&
_output_shapes
: @*
dtype0�
conv2d_transposeConv2DBackpropInputstack:output:0'conv2d_transpose/ReadVariableOp:value:0inputs*
T0*A
_output_shapes/
-:+��������������������������� *
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype0�
BiasAddBiasAddconv2d_transpose:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+��������������������������� y
IdentityIdentityBiasAdd:output:0^NoOp*
T0*A
_output_shapes/
-:+��������������������������� �
NoOpNoOp^BiasAdd/ReadVariableOp ^conv2d_transpose/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:+���������������������������@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2B
conv2d_transpose/ReadVariableOpconv2d_transpose/ReadVariableOp:i e
A
_output_shapes/
-:+���������������������������@
 
_user_specified_nameinputs
�
�
>__inference_bn3_layer_call_and_return_conditional_losses_17196

inputs%
readvariableop_resource:@'
readvariableop_1_resource:@6
(fusedbatchnormv3_readvariableop_resource:@8
*fusedbatchnormv3_readvariableop_1_resource:@
identity��AssignNewValue�AssignNewValue_1�FusedBatchNormV3/ReadVariableOp�!FusedBatchNormV3/ReadVariableOp_1�ReadVariableOp�ReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype0�
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype0�
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype0�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+���������������������������@:@:@:@:@:*
epsilon%o�:*
exponential_avg_factor%
�#<�
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(�
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(}
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+���������������������������@�
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+���������������������������@: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+���������������������������@
 
_user_specified_nameinputs
�
�
'__inference_decoder_layer_call_fn_16424

inputs
unknown:	
�
	unknown_0:	�#
	unknown_1:  
	unknown_2: 
	unknown_3: 
	unknown_4: 
	unknown_5: 
	unknown_6: #
	unknown_7:@ 
	unknown_8:@
	unknown_9:@

unknown_10:@

unknown_11:@

unknown_12:@$

unknown_13:@@

unknown_14:@

unknown_15:@

unknown_16:@

unknown_17:@

unknown_18:@$

unknown_19:@@

unknown_20:@

unknown_21:@

unknown_22:@

unknown_23:@

unknown_24:@$

unknown_25: @

unknown_26: 

unknown_27: 

unknown_28: 

unknown_29: 

unknown_30: $

unknown_31: 

unknown_32:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28
unknown_29
unknown_30
unknown_31
unknown_32*.
Tin'
%2#*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������*D
_read_only_resource_inputs&
$"	
 !"*-
config_proto

CPU

GPU 2J 8� *K
fFRD
B__inference_decoder_layer_call_and_return_conditional_losses_15940w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:���������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*j
_input_shapesY
W:���������
: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:���������

 
_user_specified_nameinputs
�
�
>__inference_bn2_layer_call_and_return_conditional_losses_15051

inputs%
readvariableop_resource:@'
readvariableop_1_resource:@6
(fusedbatchnormv3_readvariableop_resource:@8
*fusedbatchnormv3_readvariableop_1_resource:@
identity��FusedBatchNormV3/ReadVariableOp�!FusedBatchNormV3/ReadVariableOp_1�ReadVariableOp�ReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype0�
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype0�
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype0�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+���������������������������@:@:@:@:@:*
epsilon%o�:*
is_training( }
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+���������������������������@�
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+���������������������������@: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+���������������������������@
 
_user_specified_nameinputs
� 
�
K__inference_convTranspose2D1_layer_call_and_return_conditional_losses_16924

inputsB
(conv2d_transpose_readvariableop_resource:  -
biasadd_readvariableop_resource: 
identity��BiasAdd/ReadVariableOp�conv2d_transpose/ReadVariableOpI
ShapeShapeinputs*
T0*
_output_shapes
::��]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_1StridedSliceShape:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_2StridedSliceShape:output:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskG
mul/yConst*
_output_shapes
: *
dtype0*
value	B :U
mulMulstrided_slice_1:output:0mul/y:output:0*
T0*
_output_shapes
: I
mul_1/yConst*
_output_shapes
: *
dtype0*
value	B :Y
mul_1Mulstrided_slice_2:output:0mul_1/y:output:0*
T0*
_output_shapes
: I
stack/3Const*
_output_shapes
: *
dtype0*
value	B : y
stackPackstrided_slice:output:0mul:z:0	mul_1:z:0stack/3:output:0*
N*
T0*
_output_shapes
:_
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_3StridedSlicestack:output:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask�
conv2d_transpose/ReadVariableOpReadVariableOp(conv2d_transpose_readvariableop_resource*&
_output_shapes
:  *
dtype0�
conv2d_transposeConv2DBackpropInputstack:output:0'conv2d_transpose/ReadVariableOp:value:0inputs*
T0*A
_output_shapes/
-:+��������������������������� *
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype0�
BiasAddBiasAddconv2d_transpose:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+��������������������������� y
IdentityIdentityBiasAdd:output:0^NoOp*
T0*A
_output_shapes/
-:+��������������������������� �
NoOpNoOp^BiasAdd/ReadVariableOp ^conv2d_transpose/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:+��������������������������� : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2B
conv2d_transpose/ReadVariableOpconv2d_transpose/ReadVariableOp:i e
A
_output_shapes/
-:+��������������������������� 
 
_user_specified_nameinputs"�
L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*�
serving_default�
A

inputLayer3
serving_default_inputLayer:0���������
L
convTranspose2D68
StatefulPartitionedCall:0���������tensorflow/serving/predict:��
�
layer-0
layer_with_weights-0
layer-1
layer-2
layer_with_weights-1
layer-3
layer_with_weights-2
layer-4
layer-5
layer_with_weights-3
layer-6
layer_with_weights-4
layer-7
	layer-8

layer_with_weights-5

layer-9
layer_with_weights-6
layer-10
layer-11
layer_with_weights-7
layer-12
layer_with_weights-8
layer-13
layer-14
layer_with_weights-9
layer-15
layer_with_weights-10
layer-16
layer-17
layer_with_weights-11
layer-18
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
_default_save_signature

signatures"
_tf_keras_network
"
_tf_keras_input_layer
�
	variables
trainable_variables
regularization_losses
	keras_api
 __call__
*!&call_and_return_all_conditional_losses

"kernel
#bias"
_tf_keras_layer
�
$	variables
%trainable_variables
&regularization_losses
'	keras_api
(__call__
*)&call_and_return_all_conditional_losses"
_tf_keras_layer
�
*	variables
+trainable_variables
,regularization_losses
-	keras_api
.__call__
*/&call_and_return_all_conditional_losses

0kernel
1bias
 2_jit_compiled_convolution_op"
_tf_keras_layer
�
3	variables
4trainable_variables
5regularization_losses
6	keras_api
7__call__
*8&call_and_return_all_conditional_losses
9axis
	:gamma
;beta
<moving_mean
=moving_variance"
_tf_keras_layer
�
>	variables
?trainable_variables
@regularization_losses
A	keras_api
B__call__
*C&call_and_return_all_conditional_losses"
_tf_keras_layer
�
D	variables
Etrainable_variables
Fregularization_losses
G	keras_api
H__call__
*I&call_and_return_all_conditional_losses

Jkernel
Kbias
 L_jit_compiled_convolution_op"
_tf_keras_layer
�
M	variables
Ntrainable_variables
Oregularization_losses
P	keras_api
Q__call__
*R&call_and_return_all_conditional_losses
Saxis
	Tgamma
Ubeta
Vmoving_mean
Wmoving_variance"
_tf_keras_layer
�
X	variables
Ytrainable_variables
Zregularization_losses
[	keras_api
\__call__
*]&call_and_return_all_conditional_losses"
_tf_keras_layer
�
^	variables
_trainable_variables
`regularization_losses
a	keras_api
b__call__
*c&call_and_return_all_conditional_losses

dkernel
ebias
 f_jit_compiled_convolution_op"
_tf_keras_layer
�
g	variables
htrainable_variables
iregularization_losses
j	keras_api
k__call__
*l&call_and_return_all_conditional_losses
maxis
	ngamma
obeta
pmoving_mean
qmoving_variance"
_tf_keras_layer
�
r	variables
strainable_variables
tregularization_losses
u	keras_api
v__call__
*w&call_and_return_all_conditional_losses"
_tf_keras_layer
�
x	variables
ytrainable_variables
zregularization_losses
{	keras_api
|__call__
*}&call_and_return_all_conditional_losses

~kernel
bias
!�_jit_compiled_convolution_op"
_tf_keras_layer
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
	�axis

�gamma
	�beta
�moving_mean
�moving_variance"
_tf_keras_layer
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses"
_tf_keras_layer
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�kernel
	�bias
!�_jit_compiled_convolution_op"
_tf_keras_layer
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
	�axis

�gamma
	�beta
�moving_mean
�moving_variance"
_tf_keras_layer
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses"
_tf_keras_layer
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�kernel
	�bias
!�_jit_compiled_convolution_op"
_tf_keras_layer
�
"0
#1
02
13
:4
;5
<6
=7
J8
K9
T10
U11
V12
W13
d14
e15
n16
o17
p18
q19
~20
21
�22
�23
�24
�25
�26
�27
�28
�29
�30
�31
�32
�33"
trackable_list_wrapper
�
"0
#1
02
13
:4
;5
J6
K7
T8
U9
d10
e11
n12
o13
~14
15
�16
�17
�18
�19
�20
�21
�22
�23"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
	variables
trainable_variables
regularization_losses
__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
�
�trace_0
�trace_1
�trace_2
�trace_32�
'__inference_decoder_layer_call_fn_15848
'__inference_decoder_layer_call_fn_16011
'__inference_decoder_layer_call_fn_16351
'__inference_decoder_layer_call_fn_16424�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0z�trace_1z�trace_2z�trace_3
�
�trace_0
�trace_1
�trace_2
�trace_32�
B__inference_decoder_layer_call_and_return_conditional_losses_15594
B__inference_decoder_layer_call_and_return_conditional_losses_15684
B__inference_decoder_layer_call_and_return_conditional_losses_16634
B__inference_decoder_layer_call_and_return_conditional_losses_16844�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0z�trace_1z�trace_2z�trace_3
�B�
 __inference__wrapped_model_14862
inputLayer"�
���
FullArgSpec
args� 
varargsjargs
varkwjkwargs
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
-
�serving_default"
signature_map
.
"0
#1"
trackable_list_wrapper
.
"0
#1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
	variables
trainable_variables
regularization_losses
 __call__
*!&call_and_return_all_conditional_losses
&!"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
&__inference_dense1_layer_call_fn_16853�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
A__inference_dense1_layer_call_and_return_conditional_losses_16863�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
 :	
�2dense1/kernel
:�2dense1/bias
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
$	variables
%trainable_variables
&regularization_losses
(__call__
*)&call_and_return_all_conditional_losses
&)"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
,__inference_reshapeLayer_layer_call_fn_16868�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
G__inference_reshapeLayer_layer_call_and_return_conditional_losses_16882�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
.
00
11"
trackable_list_wrapper
.
00
11"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
*	variables
+trainable_variables
,regularization_losses
.__call__
*/&call_and_return_all_conditional_losses
&/"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
0__inference_convTranspose2D1_layer_call_fn_16891�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
K__inference_convTranspose2D1_layer_call_and_return_conditional_losses_16924�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
1:/  2convTranspose2D1/kernel
#:! 2convTranspose2D1/bias
�2��
���
FullArgSpec
args�
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 0
<
:0
;1
<2
=3"
trackable_list_wrapper
.
:0
;1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
3	variables
4trainable_variables
5regularization_losses
7__call__
*8&call_and_return_all_conditional_losses
&8"call_and_return_conditional_losses"
_generic_user_object
�
�trace_0
�trace_12�
#__inference_bn1_layer_call_fn_16937
#__inference_bn1_layer_call_fn_16950�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0z�trace_1
�
�trace_0
�trace_12�
>__inference_bn1_layer_call_and_return_conditional_losses_16968
>__inference_bn1_layer_call_and_return_conditional_losses_16986�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0z�trace_1
 "
trackable_list_wrapper
: 2	bn1/gamma
: 2bn1/beta
:  (2bn1/moving_mean
#:!  (2bn1/moving_variance
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
>	variables
?trainable_variables
@regularization_losses
B__call__
*C&call_and_return_all_conditional_losses
&C"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
&__inference_lReLU1_layer_call_fn_16991�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
A__inference_lReLU1_layer_call_and_return_conditional_losses_16996�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
.
J0
K1"
trackable_list_wrapper
.
J0
K1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
D	variables
Etrainable_variables
Fregularization_losses
H__call__
*I&call_and_return_all_conditional_losses
&I"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
0__inference_convTranspose2D2_layer_call_fn_17005�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
K__inference_convTranspose2D2_layer_call_and_return_conditional_losses_17038�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
1:/@ 2convTranspose2D2/kernel
#:!@2convTranspose2D2/bias
�2��
���
FullArgSpec
args�
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 0
<
T0
U1
V2
W3"
trackable_list_wrapper
.
T0
U1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
M	variables
Ntrainable_variables
Oregularization_losses
Q__call__
*R&call_and_return_all_conditional_losses
&R"call_and_return_conditional_losses"
_generic_user_object
�
�trace_0
�trace_12�
#__inference_bn2_layer_call_fn_17051
#__inference_bn2_layer_call_fn_17064�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0z�trace_1
�
�trace_0
�trace_12�
>__inference_bn2_layer_call_and_return_conditional_losses_17082
>__inference_bn2_layer_call_and_return_conditional_losses_17100�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0z�trace_1
 "
trackable_list_wrapper
:@2	bn2/gamma
:@2bn2/beta
:@ (2bn2/moving_mean
#:!@ (2bn2/moving_variance
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
X	variables
Ytrainable_variables
Zregularization_losses
\__call__
*]&call_and_return_all_conditional_losses
&]"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
&__inference_lReLU2_layer_call_fn_17105�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
A__inference_lReLU2_layer_call_and_return_conditional_losses_17110�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
.
d0
e1"
trackable_list_wrapper
.
d0
e1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
^	variables
_trainable_variables
`regularization_losses
b__call__
*c&call_and_return_all_conditional_losses
&c"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
0__inference_convTranspose2D3_layer_call_fn_17119�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
K__inference_convTranspose2D3_layer_call_and_return_conditional_losses_17152�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
1:/@@2convTranspose2D3/kernel
#:!@2convTranspose2D3/bias
�2��
���
FullArgSpec
args�
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 0
<
n0
o1
p2
q3"
trackable_list_wrapper
.
n0
o1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
g	variables
htrainable_variables
iregularization_losses
k__call__
*l&call_and_return_all_conditional_losses
&l"call_and_return_conditional_losses"
_generic_user_object
�
�trace_0
�trace_12�
#__inference_bn3_layer_call_fn_17165
#__inference_bn3_layer_call_fn_17178�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0z�trace_1
�
�trace_0
�trace_12�
>__inference_bn3_layer_call_and_return_conditional_losses_17196
>__inference_bn3_layer_call_and_return_conditional_losses_17214�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0z�trace_1
 "
trackable_list_wrapper
:@2	bn3/gamma
:@2bn3/beta
:@ (2bn3/moving_mean
#:!@ (2bn3/moving_variance
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
r	variables
strainable_variables
tregularization_losses
v__call__
*w&call_and_return_all_conditional_losses
&w"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
&__inference_lReLU3_layer_call_fn_17219�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
A__inference_lReLU3_layer_call_and_return_conditional_losses_17224�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
.
~0
1"
trackable_list_wrapper
.
~0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
x	variables
ytrainable_variables
zregularization_losses
|__call__
*}&call_and_return_all_conditional_losses
&}"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
0__inference_convTranspose2D4_layer_call_fn_17233�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
K__inference_convTranspose2D4_layer_call_and_return_conditional_losses_17266�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
1:/@@2convTranspose2D4/kernel
#:!@2convTranspose2D4/bias
�2��
���
FullArgSpec
args�
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 0
@
�0
�1
�2
�3"
trackable_list_wrapper
0
�0
�1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�
�trace_0
�trace_12�
#__inference_bn4_layer_call_fn_17279
#__inference_bn4_layer_call_fn_17292�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0z�trace_1
�
�trace_0
�trace_12�
>__inference_bn4_layer_call_and_return_conditional_losses_17310
>__inference_bn4_layer_call_and_return_conditional_losses_17328�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0z�trace_1
 "
trackable_list_wrapper
:@2	bn4/gamma
:@2bn4/beta
:@ (2bn4/moving_mean
#:!@ (2bn4/moving_variance
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
&__inference_lReLU4_layer_call_fn_17333�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
A__inference_lReLU4_layer_call_and_return_conditional_losses_17338�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
0
�0
�1"
trackable_list_wrapper
0
�0
�1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
0__inference_convTranspose2D5_layer_call_fn_17347�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
K__inference_convTranspose2D5_layer_call_and_return_conditional_losses_17380�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
1:/ @2convTranspose2D5/kernel
#:! 2convTranspose2D5/bias
�2��
���
FullArgSpec
args�
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 0
@
�0
�1
�2
�3"
trackable_list_wrapper
0
�0
�1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�
�trace_0
�trace_12�
#__inference_bn5_layer_call_fn_17393
#__inference_bn5_layer_call_fn_17406�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0z�trace_1
�
�trace_0
�trace_12�
>__inference_bn5_layer_call_and_return_conditional_losses_17424
>__inference_bn5_layer_call_and_return_conditional_losses_17442�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0z�trace_1
 "
trackable_list_wrapper
: 2	bn5/gamma
: 2bn5/beta
:  (2bn5/moving_mean
#:!  (2bn5/moving_variance
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
&__inference_lReLU5_layer_call_fn_17447�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
A__inference_lReLU5_layer_call_and_return_conditional_losses_17452�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
0
�0
�1"
trackable_list_wrapper
0
�0
�1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
0__inference_convTranspose2D6_layer_call_fn_17461�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
K__inference_convTranspose2D6_layer_call_and_return_conditional_losses_17495�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
1:/ 2convTranspose2D6/kernel
#:!2convTranspose2D6/bias
�2��
���
FullArgSpec
args�
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 0
j
<0
=1
V2
W3
p4
q5
�6
�7
�8
�9"
trackable_list_wrapper
�
0
1
2
3
4
5
6
7
	8

9
10
11
12
13
14
15
16
17
18"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
'__inference_decoder_layer_call_fn_15848
inputLayer"�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
'__inference_decoder_layer_call_fn_16011
inputLayer"�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
'__inference_decoder_layer_call_fn_16351inputs"�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
'__inference_decoder_layer_call_fn_16424inputs"�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
B__inference_decoder_layer_call_and_return_conditional_losses_15594
inputLayer"�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
B__inference_decoder_layer_call_and_return_conditional_losses_15684
inputLayer"�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
B__inference_decoder_layer_call_and_return_conditional_losses_16634inputs"�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
B__inference_decoder_layer_call_and_return_conditional_losses_16844inputs"�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
#__inference_signature_wrapper_16278
inputLayer"�
���
FullArgSpec
args� 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
&__inference_dense1_layer_call_fn_16853inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
A__inference_dense1_layer_call_and_return_conditional_losses_16863inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
,__inference_reshapeLayer_layer_call_fn_16868inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
G__inference_reshapeLayer_layer_call_and_return_conditional_losses_16882inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
0__inference_convTranspose2D1_layer_call_fn_16891inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
K__inference_convTranspose2D1_layer_call_and_return_conditional_losses_16924inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
.
<0
=1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
#__inference_bn1_layer_call_fn_16937inputs"�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
#__inference_bn1_layer_call_fn_16950inputs"�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
>__inference_bn1_layer_call_and_return_conditional_losses_16968inputs"�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
>__inference_bn1_layer_call_and_return_conditional_losses_16986inputs"�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
&__inference_lReLU1_layer_call_fn_16991inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
A__inference_lReLU1_layer_call_and_return_conditional_losses_16996inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
0__inference_convTranspose2D2_layer_call_fn_17005inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
K__inference_convTranspose2D2_layer_call_and_return_conditional_losses_17038inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
.
V0
W1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
#__inference_bn2_layer_call_fn_17051inputs"�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
#__inference_bn2_layer_call_fn_17064inputs"�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
>__inference_bn2_layer_call_and_return_conditional_losses_17082inputs"�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
>__inference_bn2_layer_call_and_return_conditional_losses_17100inputs"�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
&__inference_lReLU2_layer_call_fn_17105inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
A__inference_lReLU2_layer_call_and_return_conditional_losses_17110inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
0__inference_convTranspose2D3_layer_call_fn_17119inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
K__inference_convTranspose2D3_layer_call_and_return_conditional_losses_17152inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
.
p0
q1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
#__inference_bn3_layer_call_fn_17165inputs"�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
#__inference_bn3_layer_call_fn_17178inputs"�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
>__inference_bn3_layer_call_and_return_conditional_losses_17196inputs"�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
>__inference_bn3_layer_call_and_return_conditional_losses_17214inputs"�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
&__inference_lReLU3_layer_call_fn_17219inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
A__inference_lReLU3_layer_call_and_return_conditional_losses_17224inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
0__inference_convTranspose2D4_layer_call_fn_17233inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
K__inference_convTranspose2D4_layer_call_and_return_conditional_losses_17266inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
0
�0
�1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
#__inference_bn4_layer_call_fn_17279inputs"�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
#__inference_bn4_layer_call_fn_17292inputs"�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
>__inference_bn4_layer_call_and_return_conditional_losses_17310inputs"�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
>__inference_bn4_layer_call_and_return_conditional_losses_17328inputs"�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
&__inference_lReLU4_layer_call_fn_17333inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
A__inference_lReLU4_layer_call_and_return_conditional_losses_17338inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
0__inference_convTranspose2D5_layer_call_fn_17347inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
K__inference_convTranspose2D5_layer_call_and_return_conditional_losses_17380inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
0
�0
�1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
#__inference_bn5_layer_call_fn_17393inputs"�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
#__inference_bn5_layer_call_fn_17406inputs"�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
>__inference_bn5_layer_call_and_return_conditional_losses_17424inputs"�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
>__inference_bn5_layer_call_and_return_conditional_losses_17442inputs"�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
&__inference_lReLU5_layer_call_fn_17447inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
A__inference_lReLU5_layer_call_and_return_conditional_losses_17452inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
0__inference_convTranspose2D6_layer_call_fn_17461inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
K__inference_convTranspose2D6_layer_call_and_return_conditional_losses_17495inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 �
 __inference__wrapped_model_14862�."#01:;<=JKTUVWdenopq~������������3�0
)�&
$�!

inputLayer���������

� "K�H
F
convTranspose2D62�/
convtranspose2d6����������
>__inference_bn1_layer_call_and_return_conditional_losses_16968�:;<=Q�N
G�D
:�7
inputs+��������������������������� 
p

 
� "F�C
<�9
tensor_0+��������������������������� 
� �
>__inference_bn1_layer_call_and_return_conditional_losses_16986�:;<=Q�N
G�D
:�7
inputs+��������������������������� 
p 

 
� "F�C
<�9
tensor_0+��������������������������� 
� �
#__inference_bn1_layer_call_fn_16937�:;<=Q�N
G�D
:�7
inputs+��������������������������� 
p

 
� ";�8
unknown+��������������������������� �
#__inference_bn1_layer_call_fn_16950�:;<=Q�N
G�D
:�7
inputs+��������������������������� 
p 

 
� ";�8
unknown+��������������������������� �
>__inference_bn2_layer_call_and_return_conditional_losses_17082�TUVWQ�N
G�D
:�7
inputs+���������������������������@
p

 
� "F�C
<�9
tensor_0+���������������������������@
� �
>__inference_bn2_layer_call_and_return_conditional_losses_17100�TUVWQ�N
G�D
:�7
inputs+���������������������������@
p 

 
� "F�C
<�9
tensor_0+���������������������������@
� �
#__inference_bn2_layer_call_fn_17051�TUVWQ�N
G�D
:�7
inputs+���������������������������@
p

 
� ";�8
unknown+���������������������������@�
#__inference_bn2_layer_call_fn_17064�TUVWQ�N
G�D
:�7
inputs+���������������������������@
p 

 
� ";�8
unknown+���������������������������@�
>__inference_bn3_layer_call_and_return_conditional_losses_17196�nopqQ�N
G�D
:�7
inputs+���������������������������@
p

 
� "F�C
<�9
tensor_0+���������������������������@
� �
>__inference_bn3_layer_call_and_return_conditional_losses_17214�nopqQ�N
G�D
:�7
inputs+���������������������������@
p 

 
� "F�C
<�9
tensor_0+���������������������������@
� �
#__inference_bn3_layer_call_fn_17165�nopqQ�N
G�D
:�7
inputs+���������������������������@
p

 
� ";�8
unknown+���������������������������@�
#__inference_bn3_layer_call_fn_17178�nopqQ�N
G�D
:�7
inputs+���������������������������@
p 

 
� ";�8
unknown+���������������������������@�
>__inference_bn4_layer_call_and_return_conditional_losses_17310�����Q�N
G�D
:�7
inputs+���������������������������@
p

 
� "F�C
<�9
tensor_0+���������������������������@
� �
>__inference_bn4_layer_call_and_return_conditional_losses_17328�����Q�N
G�D
:�7
inputs+���������������������������@
p 

 
� "F�C
<�9
tensor_0+���������������������������@
� �
#__inference_bn4_layer_call_fn_17279�����Q�N
G�D
:�7
inputs+���������������������������@
p

 
� ";�8
unknown+���������������������������@�
#__inference_bn4_layer_call_fn_17292�����Q�N
G�D
:�7
inputs+���������������������������@
p 

 
� ";�8
unknown+���������������������������@�
>__inference_bn5_layer_call_and_return_conditional_losses_17424�����Q�N
G�D
:�7
inputs+��������������������������� 
p

 
� "F�C
<�9
tensor_0+��������������������������� 
� �
>__inference_bn5_layer_call_and_return_conditional_losses_17442�����Q�N
G�D
:�7
inputs+��������������������������� 
p 

 
� "F�C
<�9
tensor_0+��������������������������� 
� �
#__inference_bn5_layer_call_fn_17393�����Q�N
G�D
:�7
inputs+��������������������������� 
p

 
� ";�8
unknown+��������������������������� �
#__inference_bn5_layer_call_fn_17406�����Q�N
G�D
:�7
inputs+��������������������������� 
p 

 
� ";�8
unknown+��������������������������� �
K__inference_convTranspose2D1_layer_call_and_return_conditional_losses_16924�01I�F
?�<
:�7
inputs+��������������������������� 
� "F�C
<�9
tensor_0+��������������������������� 
� �
0__inference_convTranspose2D1_layer_call_fn_16891�01I�F
?�<
:�7
inputs+��������������������������� 
� ";�8
unknown+��������������������������� �
K__inference_convTranspose2D2_layer_call_and_return_conditional_losses_17038�JKI�F
?�<
:�7
inputs+��������������������������� 
� "F�C
<�9
tensor_0+���������������������������@
� �
0__inference_convTranspose2D2_layer_call_fn_17005�JKI�F
?�<
:�7
inputs+��������������������������� 
� ";�8
unknown+���������������������������@�
K__inference_convTranspose2D3_layer_call_and_return_conditional_losses_17152�deI�F
?�<
:�7
inputs+���������������������������@
� "F�C
<�9
tensor_0+���������������������������@
� �
0__inference_convTranspose2D3_layer_call_fn_17119�deI�F
?�<
:�7
inputs+���������������������������@
� ";�8
unknown+���������������������������@�
K__inference_convTranspose2D4_layer_call_and_return_conditional_losses_17266�~I�F
?�<
:�7
inputs+���������������������������@
� "F�C
<�9
tensor_0+���������������������������@
� �
0__inference_convTranspose2D4_layer_call_fn_17233�~I�F
?�<
:�7
inputs+���������������������������@
� ";�8
unknown+���������������������������@�
K__inference_convTranspose2D5_layer_call_and_return_conditional_losses_17380���I�F
?�<
:�7
inputs+���������������������������@
� "F�C
<�9
tensor_0+��������������������������� 
� �
0__inference_convTranspose2D5_layer_call_fn_17347���I�F
?�<
:�7
inputs+���������������������������@
� ";�8
unknown+��������������������������� �
K__inference_convTranspose2D6_layer_call_and_return_conditional_losses_17495���I�F
?�<
:�7
inputs+��������������������������� 
� "F�C
<�9
tensor_0+���������������������������
� �
0__inference_convTranspose2D6_layer_call_fn_17461���I�F
?�<
:�7
inputs+��������������������������� 
� ";�8
unknown+����������������������������
B__inference_decoder_layer_call_and_return_conditional_losses_15594�."#01:;<=JKTUVWdenopq~������������;�8
1�.
$�!

inputLayer���������

p

 
� "4�1
*�'
tensor_0���������
� �
B__inference_decoder_layer_call_and_return_conditional_losses_15684�."#01:;<=JKTUVWdenopq~������������;�8
1�.
$�!

inputLayer���������

p 

 
� "4�1
*�'
tensor_0���������
� �
B__inference_decoder_layer_call_and_return_conditional_losses_16634�."#01:;<=JKTUVWdenopq~������������7�4
-�*
 �
inputs���������

p

 
� "4�1
*�'
tensor_0���������
� �
B__inference_decoder_layer_call_and_return_conditional_losses_16844�."#01:;<=JKTUVWdenopq~������������7�4
-�*
 �
inputs���������

p 

 
� "4�1
*�'
tensor_0���������
� �
'__inference_decoder_layer_call_fn_15848�."#01:;<=JKTUVWdenopq~������������;�8
1�.
$�!

inputLayer���������

p

 
� ")�&
unknown����������
'__inference_decoder_layer_call_fn_16011�."#01:;<=JKTUVWdenopq~������������;�8
1�.
$�!

inputLayer���������

p 

 
� ")�&
unknown����������
'__inference_decoder_layer_call_fn_16351�."#01:;<=JKTUVWdenopq~������������7�4
-�*
 �
inputs���������

p

 
� ")�&
unknown����������
'__inference_decoder_layer_call_fn_16424�."#01:;<=JKTUVWdenopq~������������7�4
-�*
 �
inputs���������

p 

 
� ")�&
unknown����������
A__inference_dense1_layer_call_and_return_conditional_losses_16863d"#/�,
%�"
 �
inputs���������

� "-�*
#� 
tensor_0����������
� �
&__inference_dense1_layer_call_fn_16853Y"#/�,
%�"
 �
inputs���������

� ""�
unknown�����������
A__inference_lReLU1_layer_call_and_return_conditional_losses_16996o7�4
-�*
(�%
inputs��������� 
� "4�1
*�'
tensor_0��������� 
� �
&__inference_lReLU1_layer_call_fn_16991d7�4
-�*
(�%
inputs��������� 
� ")�&
unknown��������� �
A__inference_lReLU2_layer_call_and_return_conditional_losses_17110o7�4
-�*
(�%
inputs���������@
� "4�1
*�'
tensor_0���������@
� �
&__inference_lReLU2_layer_call_fn_17105d7�4
-�*
(�%
inputs���������@
� ")�&
unknown���������@�
A__inference_lReLU3_layer_call_and_return_conditional_losses_17224o7�4
-�*
(�%
inputs���������@
� "4�1
*�'
tensor_0���������@
� �
&__inference_lReLU3_layer_call_fn_17219d7�4
-�*
(�%
inputs���������@
� ")�&
unknown���������@�
A__inference_lReLU4_layer_call_and_return_conditional_losses_17338o7�4
-�*
(�%
inputs���������@
� "4�1
*�'
tensor_0���������@
� �
&__inference_lReLU4_layer_call_fn_17333d7�4
-�*
(�%
inputs���������@
� ")�&
unknown���������@�
A__inference_lReLU5_layer_call_and_return_conditional_losses_17452o7�4
-�*
(�%
inputs��������� 
� "4�1
*�'
tensor_0��������� 
� �
&__inference_lReLU5_layer_call_fn_17447d7�4
-�*
(�%
inputs��������� 
� ")�&
unknown��������� �
G__inference_reshapeLayer_layer_call_and_return_conditional_losses_16882h0�-
&�#
!�
inputs����������
� "4�1
*�'
tensor_0��������� 
� �
,__inference_reshapeLayer_layer_call_fn_16868]0�-
&�#
!�
inputs����������
� ")�&
unknown��������� �
#__inference_signature_wrapper_16278�."#01:;<=JKTUVWdenopq~������������A�>
� 
7�4
2

inputLayer$�!

inputlayer���������
"K�H
F
convTranspose2D62�/
convtranspose2d6���������