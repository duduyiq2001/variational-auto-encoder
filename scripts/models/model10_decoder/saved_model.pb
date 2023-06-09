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
#__inference_signature_wrapper_15736

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
__inference__traced_save_17180
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
!__inference__traced_restore_17292Į
�
]
A__inference_lReLU5_layer_call_and_return_conditional_losses_15044

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
�
�
#__inference_bn2_layer_call_fn_16509

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
>__inference_bn2_layer_call_and_return_conditional_losses_14491�
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
�R
�
B__inference_decoder_layer_call_and_return_conditional_losses_15052

inputlayer
dense1_14920:	
�
dense1_14922:	�0
convtranspose2d1_14941:  $
convtranspose2d1_14943: 
	bn1_14946: 
	bn1_14948: 
	bn1_14950: 
	bn1_14952: 0
convtranspose2d2_14962:@ $
convtranspose2d2_14964:@
	bn2_14967:@
	bn2_14969:@
	bn2_14971:@
	bn2_14973:@0
convtranspose2d3_14983:@@$
convtranspose2d3_14985:@
	bn3_14988:@
	bn3_14990:@
	bn3_14992:@
	bn3_14994:@0
convtranspose2d4_15004:@@$
convtranspose2d4_15006:@
	bn4_15009:@
	bn4_15011:@
	bn4_15013:@
	bn4_15015:@0
convtranspose2d5_15025: @$
convtranspose2d5_15027: 
	bn5_15030: 
	bn5_15032: 
	bn5_15034: 
	bn5_15036: 0
convtranspose2d6_15046: $
convtranspose2d6_15048:
identity��bn1/StatefulPartitionedCall�bn2/StatefulPartitionedCall�bn3/StatefulPartitionedCall�bn4/StatefulPartitionedCall�bn5/StatefulPartitionedCall�(convTranspose2D1/StatefulPartitionedCall�(convTranspose2D2/StatefulPartitionedCall�(convTranspose2D3/StatefulPartitionedCall�(convTranspose2D4/StatefulPartitionedCall�(convTranspose2D5/StatefulPartitionedCall�(convTranspose2D6/StatefulPartitionedCall�dense1/StatefulPartitionedCall�
dense1/StatefulPartitionedCallStatefulPartitionedCall
inputlayerdense1_14920dense1_14922*
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
A__inference_dense1_layer_call_and_return_conditional_losses_14919�
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
G__inference_reshapeLayer_layer_call_and_return_conditional_losses_14939�
(convTranspose2D1/StatefulPartitionedCallStatefulPartitionedCall%reshapeLayer/PartitionedCall:output:0convtranspose2d1_14941convtranspose2d1_14943*
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
K__inference_convTranspose2D1_layer_call_and_return_conditional_losses_14354�
bn1/StatefulPartitionedCallStatefulPartitionedCall1convTranspose2D1/StatefulPartitionedCall:output:0	bn1_14946	bn1_14948	bn1_14950	bn1_14952*
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
>__inference_bn1_layer_call_and_return_conditional_losses_14383�
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
A__inference_lReLU1_layer_call_and_return_conditional_losses_14960�
(convTranspose2D2/StatefulPartitionedCallStatefulPartitionedCalllReLU1/PartitionedCall:output:0convtranspose2d2_14962convtranspose2d2_14964*
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
K__inference_convTranspose2D2_layer_call_and_return_conditional_losses_14462�
bn2/StatefulPartitionedCallStatefulPartitionedCall1convTranspose2D2/StatefulPartitionedCall:output:0	bn2_14967	bn2_14969	bn2_14971	bn2_14973*
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
>__inference_bn2_layer_call_and_return_conditional_losses_14491�
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
A__inference_lReLU2_layer_call_and_return_conditional_losses_14981�
(convTranspose2D3/StatefulPartitionedCallStatefulPartitionedCalllReLU2/PartitionedCall:output:0convtranspose2d3_14983convtranspose2d3_14985*
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
K__inference_convTranspose2D3_layer_call_and_return_conditional_losses_14570�
bn3/StatefulPartitionedCallStatefulPartitionedCall1convTranspose2D3/StatefulPartitionedCall:output:0	bn3_14988	bn3_14990	bn3_14992	bn3_14994*
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
>__inference_bn3_layer_call_and_return_conditional_losses_14599�
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
A__inference_lReLU3_layer_call_and_return_conditional_losses_15002�
(convTranspose2D4/StatefulPartitionedCallStatefulPartitionedCalllReLU3/PartitionedCall:output:0convtranspose2d4_15004convtranspose2d4_15006*
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
K__inference_convTranspose2D4_layer_call_and_return_conditional_losses_14678�
bn4/StatefulPartitionedCallStatefulPartitionedCall1convTranspose2D4/StatefulPartitionedCall:output:0	bn4_15009	bn4_15011	bn4_15013	bn4_15015*
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
>__inference_bn4_layer_call_and_return_conditional_losses_14707�
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
A__inference_lReLU4_layer_call_and_return_conditional_losses_15023�
(convTranspose2D5/StatefulPartitionedCallStatefulPartitionedCalllReLU4/PartitionedCall:output:0convtranspose2d5_15025convtranspose2d5_15027*
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
K__inference_convTranspose2D5_layer_call_and_return_conditional_losses_14786�
bn5/StatefulPartitionedCallStatefulPartitionedCall1convTranspose2D5/StatefulPartitionedCall:output:0	bn5_15030	bn5_15032	bn5_15034	bn5_15036*
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
>__inference_bn5_layer_call_and_return_conditional_losses_14815�
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
A__inference_lReLU5_layer_call_and_return_conditional_losses_15044�
(convTranspose2D6/StatefulPartitionedCallStatefulPartitionedCalllReLU5/PartitionedCall:output:0convtranspose2d6_15046convtranspose2d6_15048*
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
K__inference_convTranspose2D6_layer_call_and_return_conditional_losses_14895�
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
�
�
#__inference_signature_wrapper_15736

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
 __inference__wrapped_model_14320w
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
A__inference_lReLU2_layer_call_and_return_conditional_losses_16568

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
� 
�
K__inference_convTranspose2D3_layer_call_and_return_conditional_losses_16610

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
�
�
#__inference_bn3_layer_call_fn_16623

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
>__inference_bn3_layer_call_and_return_conditional_losses_14599�
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
�
�
>__inference_bn5_layer_call_and_return_conditional_losses_14833

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
�
�
>__inference_bn3_layer_call_and_return_conditional_losses_14617

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
>__inference_bn3_layer_call_and_return_conditional_losses_14599

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
K__inference_convTranspose2D5_layer_call_and_return_conditional_losses_14786

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
0__inference_convTranspose2D4_layer_call_fn_16691

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
K__inference_convTranspose2D4_layer_call_and_return_conditional_losses_14678�
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
�
]
A__inference_lReLU5_layer_call_and_return_conditional_losses_16910

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
�
�
'__inference_decoder_layer_call_fn_15809

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
B__inference_decoder_layer_call_and_return_conditional_losses_15235w
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
>__inference_bn2_layer_call_and_return_conditional_losses_16558

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
>__inference_bn3_layer_call_and_return_conditional_losses_16672

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
&__inference_lReLU1_layer_call_fn_16449

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
A__inference_lReLU1_layer_call_and_return_conditional_losses_14960h
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
>__inference_bn2_layer_call_and_return_conditional_losses_14509

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
>__inference_bn1_layer_call_and_return_conditional_losses_16444

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
�
]
A__inference_lReLU4_layer_call_and_return_conditional_losses_15023

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
�
�
'__inference_decoder_layer_call_fn_15469

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
B__inference_decoder_layer_call_and_return_conditional_losses_15398w
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
0__inference_convTranspose2D2_layer_call_fn_16463

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
K__inference_convTranspose2D2_layer_call_and_return_conditional_losses_14462�
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
�
A__inference_dense1_layer_call_and_return_conditional_losses_16321

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
��
�
 __inference__wrapped_model_14320

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
�
�
#__inference_bn3_layer_call_fn_16636

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
>__inference_bn3_layer_call_and_return_conditional_losses_14617�
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
K__inference_convTranspose2D1_layer_call_and_return_conditional_losses_14354

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
�
�
>__inference_bn5_layer_call_and_return_conditional_losses_16882

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
�
�
#__inference_bn2_layer_call_fn_16522

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
>__inference_bn2_layer_call_and_return_conditional_losses_14509�
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
�R
�
B__inference_decoder_layer_call_and_return_conditional_losses_15235

inputs
dense1_15148:	
�
dense1_15150:	�0
convtranspose2d1_15154:  $
convtranspose2d1_15156: 
	bn1_15159: 
	bn1_15161: 
	bn1_15163: 
	bn1_15165: 0
convtranspose2d2_15169:@ $
convtranspose2d2_15171:@
	bn2_15174:@
	bn2_15176:@
	bn2_15178:@
	bn2_15180:@0
convtranspose2d3_15184:@@$
convtranspose2d3_15186:@
	bn3_15189:@
	bn3_15191:@
	bn3_15193:@
	bn3_15195:@0
convtranspose2d4_15199:@@$
convtranspose2d4_15201:@
	bn4_15204:@
	bn4_15206:@
	bn4_15208:@
	bn4_15210:@0
convtranspose2d5_15214: @$
convtranspose2d5_15216: 
	bn5_15219: 
	bn5_15221: 
	bn5_15223: 
	bn5_15225: 0
convtranspose2d6_15229: $
convtranspose2d6_15231:
identity��bn1/StatefulPartitionedCall�bn2/StatefulPartitionedCall�bn3/StatefulPartitionedCall�bn4/StatefulPartitionedCall�bn5/StatefulPartitionedCall�(convTranspose2D1/StatefulPartitionedCall�(convTranspose2D2/StatefulPartitionedCall�(convTranspose2D3/StatefulPartitionedCall�(convTranspose2D4/StatefulPartitionedCall�(convTranspose2D5/StatefulPartitionedCall�(convTranspose2D6/StatefulPartitionedCall�dense1/StatefulPartitionedCall�
dense1/StatefulPartitionedCallStatefulPartitionedCallinputsdense1_15148dense1_15150*
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
A__inference_dense1_layer_call_and_return_conditional_losses_14919�
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
G__inference_reshapeLayer_layer_call_and_return_conditional_losses_14939�
(convTranspose2D1/StatefulPartitionedCallStatefulPartitionedCall%reshapeLayer/PartitionedCall:output:0convtranspose2d1_15154convtranspose2d1_15156*
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
K__inference_convTranspose2D1_layer_call_and_return_conditional_losses_14354�
bn1/StatefulPartitionedCallStatefulPartitionedCall1convTranspose2D1/StatefulPartitionedCall:output:0	bn1_15159	bn1_15161	bn1_15163	bn1_15165*
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
>__inference_bn1_layer_call_and_return_conditional_losses_14383�
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
A__inference_lReLU1_layer_call_and_return_conditional_losses_14960�
(convTranspose2D2/StatefulPartitionedCallStatefulPartitionedCalllReLU1/PartitionedCall:output:0convtranspose2d2_15169convtranspose2d2_15171*
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
K__inference_convTranspose2D2_layer_call_and_return_conditional_losses_14462�
bn2/StatefulPartitionedCallStatefulPartitionedCall1convTranspose2D2/StatefulPartitionedCall:output:0	bn2_15174	bn2_15176	bn2_15178	bn2_15180*
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
>__inference_bn2_layer_call_and_return_conditional_losses_14491�
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
A__inference_lReLU2_layer_call_and_return_conditional_losses_14981�
(convTranspose2D3/StatefulPartitionedCallStatefulPartitionedCalllReLU2/PartitionedCall:output:0convtranspose2d3_15184convtranspose2d3_15186*
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
K__inference_convTranspose2D3_layer_call_and_return_conditional_losses_14570�
bn3/StatefulPartitionedCallStatefulPartitionedCall1convTranspose2D3/StatefulPartitionedCall:output:0	bn3_15189	bn3_15191	bn3_15193	bn3_15195*
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
>__inference_bn3_layer_call_and_return_conditional_losses_14599�
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
A__inference_lReLU3_layer_call_and_return_conditional_losses_15002�
(convTranspose2D4/StatefulPartitionedCallStatefulPartitionedCalllReLU3/PartitionedCall:output:0convtranspose2d4_15199convtranspose2d4_15201*
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
K__inference_convTranspose2D4_layer_call_and_return_conditional_losses_14678�
bn4/StatefulPartitionedCallStatefulPartitionedCall1convTranspose2D4/StatefulPartitionedCall:output:0	bn4_15204	bn4_15206	bn4_15208	bn4_15210*
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
>__inference_bn4_layer_call_and_return_conditional_losses_14707�
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
A__inference_lReLU4_layer_call_and_return_conditional_losses_15023�
(convTranspose2D5/StatefulPartitionedCallStatefulPartitionedCalllReLU4/PartitionedCall:output:0convtranspose2d5_15214convtranspose2d5_15216*
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
K__inference_convTranspose2D5_layer_call_and_return_conditional_losses_14786�
bn5/StatefulPartitionedCallStatefulPartitionedCall1convTranspose2D5/StatefulPartitionedCall:output:0	bn5_15219	bn5_15221	bn5_15223	bn5_15225*
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
>__inference_bn5_layer_call_and_return_conditional_losses_14815�
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
A__inference_lReLU5_layer_call_and_return_conditional_losses_15044�
(convTranspose2D6/StatefulPartitionedCallStatefulPartitionedCalllReLU5/PartitionedCall:output:0convtranspose2d6_15229convtranspose2d6_15231*
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
K__inference_convTranspose2D6_layer_call_and_return_conditional_losses_14895�
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
�
�
>__inference_bn2_layer_call_and_return_conditional_losses_14491

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
�
�
&__inference_dense1_layer_call_fn_16311

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
A__inference_dense1_layer_call_and_return_conditional_losses_14919p
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
�
�
#__inference_bn1_layer_call_fn_16395

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
>__inference_bn1_layer_call_and_return_conditional_losses_14383�
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
ؐ
�
!__inference__traced_restore_17292
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
�!
�
K__inference_convTranspose2D6_layer_call_and_return_conditional_losses_16953

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
#__inference_bn1_layer_call_fn_16408

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
>__inference_bn1_layer_call_and_return_conditional_losses_14401�
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
� 
�
K__inference_convTranspose2D1_layer_call_and_return_conditional_losses_16382

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
�
�
>__inference_bn2_layer_call_and_return_conditional_losses_16540

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
c
G__inference_reshapeLayer_layer_call_and_return_conditional_losses_14939

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
�
�
>__inference_bn1_layer_call_and_return_conditional_losses_14383

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
K__inference_convTranspose2D5_layer_call_and_return_conditional_losses_16838

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
�
�
>__inference_bn4_layer_call_and_return_conditional_losses_14725

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
K__inference_convTranspose2D4_layer_call_and_return_conditional_losses_16724

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
>__inference_bn4_layer_call_and_return_conditional_losses_14707

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
K__inference_convTranspose2D2_layer_call_and_return_conditional_losses_16496

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
�
B
&__inference_lReLU4_layer_call_fn_16791

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
A__inference_lReLU4_layer_call_and_return_conditional_losses_15023h
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
� 
�
K__inference_convTranspose2D2_layer_call_and_return_conditional_losses_14462

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
A__inference_lReLU1_layer_call_and_return_conditional_losses_14960

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
�
�
'__inference_decoder_layer_call_fn_15306

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
B__inference_decoder_layer_call_and_return_conditional_losses_15235w
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
A__inference_lReLU3_layer_call_and_return_conditional_losses_16682

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
�
�
>__inference_bn4_layer_call_and_return_conditional_losses_16786

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
B__inference_decoder_layer_call_and_return_conditional_losses_15142

inputlayer
dense1_15055:	
�
dense1_15057:	�0
convtranspose2d1_15061:  $
convtranspose2d1_15063: 
	bn1_15066: 
	bn1_15068: 
	bn1_15070: 
	bn1_15072: 0
convtranspose2d2_15076:@ $
convtranspose2d2_15078:@
	bn2_15081:@
	bn2_15083:@
	bn2_15085:@
	bn2_15087:@0
convtranspose2d3_15091:@@$
convtranspose2d3_15093:@
	bn3_15096:@
	bn3_15098:@
	bn3_15100:@
	bn3_15102:@0
convtranspose2d4_15106:@@$
convtranspose2d4_15108:@
	bn4_15111:@
	bn4_15113:@
	bn4_15115:@
	bn4_15117:@0
convtranspose2d5_15121: @$
convtranspose2d5_15123: 
	bn5_15126: 
	bn5_15128: 
	bn5_15130: 
	bn5_15132: 0
convtranspose2d6_15136: $
convtranspose2d6_15138:
identity��bn1/StatefulPartitionedCall�bn2/StatefulPartitionedCall�bn3/StatefulPartitionedCall�bn4/StatefulPartitionedCall�bn5/StatefulPartitionedCall�(convTranspose2D1/StatefulPartitionedCall�(convTranspose2D2/StatefulPartitionedCall�(convTranspose2D3/StatefulPartitionedCall�(convTranspose2D4/StatefulPartitionedCall�(convTranspose2D5/StatefulPartitionedCall�(convTranspose2D6/StatefulPartitionedCall�dense1/StatefulPartitionedCall�
dense1/StatefulPartitionedCallStatefulPartitionedCall
inputlayerdense1_15055dense1_15057*
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
A__inference_dense1_layer_call_and_return_conditional_losses_14919�
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
G__inference_reshapeLayer_layer_call_and_return_conditional_losses_14939�
(convTranspose2D1/StatefulPartitionedCallStatefulPartitionedCall%reshapeLayer/PartitionedCall:output:0convtranspose2d1_15061convtranspose2d1_15063*
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
K__inference_convTranspose2D1_layer_call_and_return_conditional_losses_14354�
bn1/StatefulPartitionedCallStatefulPartitionedCall1convTranspose2D1/StatefulPartitionedCall:output:0	bn1_15066	bn1_15068	bn1_15070	bn1_15072*
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
>__inference_bn1_layer_call_and_return_conditional_losses_14401�
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
A__inference_lReLU1_layer_call_and_return_conditional_losses_14960�
(convTranspose2D2/StatefulPartitionedCallStatefulPartitionedCalllReLU1/PartitionedCall:output:0convtranspose2d2_15076convtranspose2d2_15078*
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
K__inference_convTranspose2D2_layer_call_and_return_conditional_losses_14462�
bn2/StatefulPartitionedCallStatefulPartitionedCall1convTranspose2D2/StatefulPartitionedCall:output:0	bn2_15081	bn2_15083	bn2_15085	bn2_15087*
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
>__inference_bn2_layer_call_and_return_conditional_losses_14509�
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
A__inference_lReLU2_layer_call_and_return_conditional_losses_14981�
(convTranspose2D3/StatefulPartitionedCallStatefulPartitionedCalllReLU2/PartitionedCall:output:0convtranspose2d3_15091convtranspose2d3_15093*
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
K__inference_convTranspose2D3_layer_call_and_return_conditional_losses_14570�
bn3/StatefulPartitionedCallStatefulPartitionedCall1convTranspose2D3/StatefulPartitionedCall:output:0	bn3_15096	bn3_15098	bn3_15100	bn3_15102*
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
>__inference_bn3_layer_call_and_return_conditional_losses_14617�
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
A__inference_lReLU3_layer_call_and_return_conditional_losses_15002�
(convTranspose2D4/StatefulPartitionedCallStatefulPartitionedCalllReLU3/PartitionedCall:output:0convtranspose2d4_15106convtranspose2d4_15108*
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
K__inference_convTranspose2D4_layer_call_and_return_conditional_losses_14678�
bn4/StatefulPartitionedCallStatefulPartitionedCall1convTranspose2D4/StatefulPartitionedCall:output:0	bn4_15111	bn4_15113	bn4_15115	bn4_15117*
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
>__inference_bn4_layer_call_and_return_conditional_losses_14725�
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
A__inference_lReLU4_layer_call_and_return_conditional_losses_15023�
(convTranspose2D5/StatefulPartitionedCallStatefulPartitionedCalllReLU4/PartitionedCall:output:0convtranspose2d5_15121convtranspose2d5_15123*
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
K__inference_convTranspose2D5_layer_call_and_return_conditional_losses_14786�
bn5/StatefulPartitionedCallStatefulPartitionedCall1convTranspose2D5/StatefulPartitionedCall:output:0	bn5_15126	bn5_15128	bn5_15130	bn5_15132*
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
>__inference_bn5_layer_call_and_return_conditional_losses_14833�
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
A__inference_lReLU5_layer_call_and_return_conditional_losses_15044�
(convTranspose2D6/StatefulPartitionedCallStatefulPartitionedCalllReLU5/PartitionedCall:output:0convtranspose2d6_15136convtranspose2d6_15138*
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
K__inference_convTranspose2D6_layer_call_and_return_conditional_losses_14895�
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
c
G__inference_reshapeLayer_layer_call_and_return_conditional_losses_16340

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
�
�
0__inference_convTranspose2D6_layer_call_fn_16919

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
K__inference_convTranspose2D6_layer_call_and_return_conditional_losses_14895�
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
�
�
#__inference_bn5_layer_call_fn_16864

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
>__inference_bn5_layer_call_and_return_conditional_losses_14833�
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
A__inference_lReLU2_layer_call_and_return_conditional_losses_14981

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
K__inference_convTranspose2D6_layer_call_and_return_conditional_losses_14895

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
A__inference_lReLU4_layer_call_and_return_conditional_losses_16796

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
�
�
>__inference_bn5_layer_call_and_return_conditional_losses_16900

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
>__inference_bn3_layer_call_and_return_conditional_losses_16654

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
#__inference_bn4_layer_call_fn_16737

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
>__inference_bn4_layer_call_and_return_conditional_losses_14707�
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
0__inference_convTranspose2D5_layer_call_fn_16805

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
K__inference_convTranspose2D5_layer_call_and_return_conditional_losses_14786�
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
0__inference_convTranspose2D1_layer_call_fn_16349

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
K__inference_convTranspose2D1_layer_call_and_return_conditional_losses_14354�
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
�
�
#__inference_bn4_layer_call_fn_16750

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
>__inference_bn4_layer_call_and_return_conditional_losses_14725�
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
K__inference_convTranspose2D4_layer_call_and_return_conditional_losses_14678

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
�
�
0__inference_convTranspose2D3_layer_call_fn_16577

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
K__inference_convTranspose2D3_layer_call_and_return_conditional_losses_14570�
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
B__inference_decoder_layer_call_and_return_conditional_losses_16092

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
#__inference_bn5_layer_call_fn_16851

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
>__inference_bn5_layer_call_and_return_conditional_losses_14815�
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
&__inference_lReLU3_layer_call_fn_16677

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
A__inference_lReLU3_layer_call_and_return_conditional_losses_15002h
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
�
H
,__inference_reshapeLayer_layer_call_fn_16326

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
G__inference_reshapeLayer_layer_call_and_return_conditional_losses_14939h
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
�
�
>__inference_bn4_layer_call_and_return_conditional_losses_16768

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
�
A__inference_dense1_layer_call_and_return_conditional_losses_14919

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
��
�
__inference__traced_save_17180
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
��
�
B__inference_decoder_layer_call_and_return_conditional_losses_16302

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
�
B
&__inference_lReLU5_layer_call_fn_16905

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
A__inference_lReLU5_layer_call_and_return_conditional_losses_15044h
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
�
B
&__inference_lReLU2_layer_call_fn_16563

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
A__inference_lReLU2_layer_call_and_return_conditional_losses_14981h
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
�
]
A__inference_lReLU1_layer_call_and_return_conditional_losses_16454

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
�
�
>__inference_bn5_layer_call_and_return_conditional_losses_14815

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
>__inference_bn1_layer_call_and_return_conditional_losses_16426

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
�
�
'__inference_decoder_layer_call_fn_15882

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
B__inference_decoder_layer_call_and_return_conditional_losses_15398w
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
�R
�
B__inference_decoder_layer_call_and_return_conditional_losses_15398

inputs
dense1_15311:	
�
dense1_15313:	�0
convtranspose2d1_15317:  $
convtranspose2d1_15319: 
	bn1_15322: 
	bn1_15324: 
	bn1_15326: 
	bn1_15328: 0
convtranspose2d2_15332:@ $
convtranspose2d2_15334:@
	bn2_15337:@
	bn2_15339:@
	bn2_15341:@
	bn2_15343:@0
convtranspose2d3_15347:@@$
convtranspose2d3_15349:@
	bn3_15352:@
	bn3_15354:@
	bn3_15356:@
	bn3_15358:@0
convtranspose2d4_15362:@@$
convtranspose2d4_15364:@
	bn4_15367:@
	bn4_15369:@
	bn4_15371:@
	bn4_15373:@0
convtranspose2d5_15377: @$
convtranspose2d5_15379: 
	bn5_15382: 
	bn5_15384: 
	bn5_15386: 
	bn5_15388: 0
convtranspose2d6_15392: $
convtranspose2d6_15394:
identity��bn1/StatefulPartitionedCall�bn2/StatefulPartitionedCall�bn3/StatefulPartitionedCall�bn4/StatefulPartitionedCall�bn5/StatefulPartitionedCall�(convTranspose2D1/StatefulPartitionedCall�(convTranspose2D2/StatefulPartitionedCall�(convTranspose2D3/StatefulPartitionedCall�(convTranspose2D4/StatefulPartitionedCall�(convTranspose2D5/StatefulPartitionedCall�(convTranspose2D6/StatefulPartitionedCall�dense1/StatefulPartitionedCall�
dense1/StatefulPartitionedCallStatefulPartitionedCallinputsdense1_15311dense1_15313*
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
A__inference_dense1_layer_call_and_return_conditional_losses_14919�
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
G__inference_reshapeLayer_layer_call_and_return_conditional_losses_14939�
(convTranspose2D1/StatefulPartitionedCallStatefulPartitionedCall%reshapeLayer/PartitionedCall:output:0convtranspose2d1_15317convtranspose2d1_15319*
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
K__inference_convTranspose2D1_layer_call_and_return_conditional_losses_14354�
bn1/StatefulPartitionedCallStatefulPartitionedCall1convTranspose2D1/StatefulPartitionedCall:output:0	bn1_15322	bn1_15324	bn1_15326	bn1_15328*
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
>__inference_bn1_layer_call_and_return_conditional_losses_14401�
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
A__inference_lReLU1_layer_call_and_return_conditional_losses_14960�
(convTranspose2D2/StatefulPartitionedCallStatefulPartitionedCalllReLU1/PartitionedCall:output:0convtranspose2d2_15332convtranspose2d2_15334*
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
K__inference_convTranspose2D2_layer_call_and_return_conditional_losses_14462�
bn2/StatefulPartitionedCallStatefulPartitionedCall1convTranspose2D2/StatefulPartitionedCall:output:0	bn2_15337	bn2_15339	bn2_15341	bn2_15343*
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
>__inference_bn2_layer_call_and_return_conditional_losses_14509�
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
A__inference_lReLU2_layer_call_and_return_conditional_losses_14981�
(convTranspose2D3/StatefulPartitionedCallStatefulPartitionedCalllReLU2/PartitionedCall:output:0convtranspose2d3_15347convtranspose2d3_15349*
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
K__inference_convTranspose2D3_layer_call_and_return_conditional_losses_14570�
bn3/StatefulPartitionedCallStatefulPartitionedCall1convTranspose2D3/StatefulPartitionedCall:output:0	bn3_15352	bn3_15354	bn3_15356	bn3_15358*
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
>__inference_bn3_layer_call_and_return_conditional_losses_14617�
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
A__inference_lReLU3_layer_call_and_return_conditional_losses_15002�
(convTranspose2D4/StatefulPartitionedCallStatefulPartitionedCalllReLU3/PartitionedCall:output:0convtranspose2d4_15362convtranspose2d4_15364*
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
K__inference_convTranspose2D4_layer_call_and_return_conditional_losses_14678�
bn4/StatefulPartitionedCallStatefulPartitionedCall1convTranspose2D4/StatefulPartitionedCall:output:0	bn4_15367	bn4_15369	bn4_15371	bn4_15373*
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
>__inference_bn4_layer_call_and_return_conditional_losses_14725�
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
A__inference_lReLU4_layer_call_and_return_conditional_losses_15023�
(convTranspose2D5/StatefulPartitionedCallStatefulPartitionedCalllReLU4/PartitionedCall:output:0convtranspose2d5_15377convtranspose2d5_15379*
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
K__inference_convTranspose2D5_layer_call_and_return_conditional_losses_14786�
bn5/StatefulPartitionedCallStatefulPartitionedCall1convTranspose2D5/StatefulPartitionedCall:output:0	bn5_15382	bn5_15384	bn5_15386	bn5_15388*
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
>__inference_bn5_layer_call_and_return_conditional_losses_14833�
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
A__inference_lReLU5_layer_call_and_return_conditional_losses_15044�
(convTranspose2D6/StatefulPartitionedCallStatefulPartitionedCalllReLU5/PartitionedCall:output:0convtranspose2d6_15392convtranspose2d6_15394*
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
K__inference_convTranspose2D6_layer_call_and_return_conditional_losses_14895�
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
� 
�
K__inference_convTranspose2D3_layer_call_and_return_conditional_losses_14570

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
�
]
A__inference_lReLU3_layer_call_and_return_conditional_losses_15002

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
�
�
>__inference_bn1_layer_call_and_return_conditional_losses_14401

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
'__inference_decoder_layer_call_fn_15306
'__inference_decoder_layer_call_fn_15469
'__inference_decoder_layer_call_fn_15809
'__inference_decoder_layer_call_fn_15882�
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
B__inference_decoder_layer_call_and_return_conditional_losses_15052
B__inference_decoder_layer_call_and_return_conditional_losses_15142
B__inference_decoder_layer_call_and_return_conditional_losses_16092
B__inference_decoder_layer_call_and_return_conditional_losses_16302�
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
 __inference__wrapped_model_14320
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
&__inference_dense1_layer_call_fn_16311�
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
A__inference_dense1_layer_call_and_return_conditional_losses_16321�
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
,__inference_reshapeLayer_layer_call_fn_16326�
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
G__inference_reshapeLayer_layer_call_and_return_conditional_losses_16340�
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
0__inference_convTranspose2D1_layer_call_fn_16349�
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
K__inference_convTranspose2D1_layer_call_and_return_conditional_losses_16382�
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
#__inference_bn1_layer_call_fn_16395
#__inference_bn1_layer_call_fn_16408�
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
>__inference_bn1_layer_call_and_return_conditional_losses_16426
>__inference_bn1_layer_call_and_return_conditional_losses_16444�
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
&__inference_lReLU1_layer_call_fn_16449�
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
A__inference_lReLU1_layer_call_and_return_conditional_losses_16454�
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
0__inference_convTranspose2D2_layer_call_fn_16463�
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
K__inference_convTranspose2D2_layer_call_and_return_conditional_losses_16496�
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
#__inference_bn2_layer_call_fn_16509
#__inference_bn2_layer_call_fn_16522�
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
>__inference_bn2_layer_call_and_return_conditional_losses_16540
>__inference_bn2_layer_call_and_return_conditional_losses_16558�
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
&__inference_lReLU2_layer_call_fn_16563�
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
A__inference_lReLU2_layer_call_and_return_conditional_losses_16568�
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
0__inference_convTranspose2D3_layer_call_fn_16577�
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
K__inference_convTranspose2D3_layer_call_and_return_conditional_losses_16610�
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
#__inference_bn3_layer_call_fn_16623
#__inference_bn3_layer_call_fn_16636�
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
>__inference_bn3_layer_call_and_return_conditional_losses_16654
>__inference_bn3_layer_call_and_return_conditional_losses_16672�
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
&__inference_lReLU3_layer_call_fn_16677�
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
A__inference_lReLU3_layer_call_and_return_conditional_losses_16682�
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
0__inference_convTranspose2D4_layer_call_fn_16691�
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
K__inference_convTranspose2D4_layer_call_and_return_conditional_losses_16724�
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
#__inference_bn4_layer_call_fn_16737
#__inference_bn4_layer_call_fn_16750�
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
>__inference_bn4_layer_call_and_return_conditional_losses_16768
>__inference_bn4_layer_call_and_return_conditional_losses_16786�
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
&__inference_lReLU4_layer_call_fn_16791�
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
A__inference_lReLU4_layer_call_and_return_conditional_losses_16796�
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
0__inference_convTranspose2D5_layer_call_fn_16805�
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
K__inference_convTranspose2D5_layer_call_and_return_conditional_losses_16838�
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
#__inference_bn5_layer_call_fn_16851
#__inference_bn5_layer_call_fn_16864�
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
>__inference_bn5_layer_call_and_return_conditional_losses_16882
>__inference_bn5_layer_call_and_return_conditional_losses_16900�
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
&__inference_lReLU5_layer_call_fn_16905�
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
A__inference_lReLU5_layer_call_and_return_conditional_losses_16910�
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
0__inference_convTranspose2D6_layer_call_fn_16919�
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
K__inference_convTranspose2D6_layer_call_and_return_conditional_losses_16953�
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
'__inference_decoder_layer_call_fn_15306
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
'__inference_decoder_layer_call_fn_15469
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
'__inference_decoder_layer_call_fn_15809inputs"�
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
'__inference_decoder_layer_call_fn_15882inputs"�
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
B__inference_decoder_layer_call_and_return_conditional_losses_15052
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
B__inference_decoder_layer_call_and_return_conditional_losses_15142
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
B__inference_decoder_layer_call_and_return_conditional_losses_16092inputs"�
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
B__inference_decoder_layer_call_and_return_conditional_losses_16302inputs"�
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
#__inference_signature_wrapper_15736
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
&__inference_dense1_layer_call_fn_16311inputs"�
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
A__inference_dense1_layer_call_and_return_conditional_losses_16321inputs"�
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
,__inference_reshapeLayer_layer_call_fn_16326inputs"�
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
G__inference_reshapeLayer_layer_call_and_return_conditional_losses_16340inputs"�
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
0__inference_convTranspose2D1_layer_call_fn_16349inputs"�
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
K__inference_convTranspose2D1_layer_call_and_return_conditional_losses_16382inputs"�
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
#__inference_bn1_layer_call_fn_16395inputs"�
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
#__inference_bn1_layer_call_fn_16408inputs"�
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
>__inference_bn1_layer_call_and_return_conditional_losses_16426inputs"�
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
>__inference_bn1_layer_call_and_return_conditional_losses_16444inputs"�
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
&__inference_lReLU1_layer_call_fn_16449inputs"�
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
A__inference_lReLU1_layer_call_and_return_conditional_losses_16454inputs"�
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
0__inference_convTranspose2D2_layer_call_fn_16463inputs"�
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
K__inference_convTranspose2D2_layer_call_and_return_conditional_losses_16496inputs"�
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
#__inference_bn2_layer_call_fn_16509inputs"�
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
#__inference_bn2_layer_call_fn_16522inputs"�
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
>__inference_bn2_layer_call_and_return_conditional_losses_16540inputs"�
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
>__inference_bn2_layer_call_and_return_conditional_losses_16558inputs"�
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
&__inference_lReLU2_layer_call_fn_16563inputs"�
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
A__inference_lReLU2_layer_call_and_return_conditional_losses_16568inputs"�
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
0__inference_convTranspose2D3_layer_call_fn_16577inputs"�
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
K__inference_convTranspose2D3_layer_call_and_return_conditional_losses_16610inputs"�
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
#__inference_bn3_layer_call_fn_16623inputs"�
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
#__inference_bn3_layer_call_fn_16636inputs"�
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
>__inference_bn3_layer_call_and_return_conditional_losses_16654inputs"�
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
>__inference_bn3_layer_call_and_return_conditional_losses_16672inputs"�
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
&__inference_lReLU3_layer_call_fn_16677inputs"�
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
A__inference_lReLU3_layer_call_and_return_conditional_losses_16682inputs"�
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
0__inference_convTranspose2D4_layer_call_fn_16691inputs"�
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
K__inference_convTranspose2D4_layer_call_and_return_conditional_losses_16724inputs"�
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
#__inference_bn4_layer_call_fn_16737inputs"�
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
#__inference_bn4_layer_call_fn_16750inputs"�
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
>__inference_bn4_layer_call_and_return_conditional_losses_16768inputs"�
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
>__inference_bn4_layer_call_and_return_conditional_losses_16786inputs"�
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
&__inference_lReLU4_layer_call_fn_16791inputs"�
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
A__inference_lReLU4_layer_call_and_return_conditional_losses_16796inputs"�
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
0__inference_convTranspose2D5_layer_call_fn_16805inputs"�
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
K__inference_convTranspose2D5_layer_call_and_return_conditional_losses_16838inputs"�
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
#__inference_bn5_layer_call_fn_16851inputs"�
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
#__inference_bn5_layer_call_fn_16864inputs"�
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
>__inference_bn5_layer_call_and_return_conditional_losses_16882inputs"�
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
>__inference_bn5_layer_call_and_return_conditional_losses_16900inputs"�
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
&__inference_lReLU5_layer_call_fn_16905inputs"�
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
A__inference_lReLU5_layer_call_and_return_conditional_losses_16910inputs"�
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
0__inference_convTranspose2D6_layer_call_fn_16919inputs"�
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
K__inference_convTranspose2D6_layer_call_and_return_conditional_losses_16953inputs"�
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
 __inference__wrapped_model_14320�."#01:;<=JKTUVWdenopq~������������3�0
)�&
$�!

inputLayer���������

� "K�H
F
convTranspose2D62�/
convtranspose2d6����������
>__inference_bn1_layer_call_and_return_conditional_losses_16426�:;<=Q�N
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
>__inference_bn1_layer_call_and_return_conditional_losses_16444�:;<=Q�N
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
#__inference_bn1_layer_call_fn_16395�:;<=Q�N
G�D
:�7
inputs+��������������������������� 
p

 
� ";�8
unknown+��������������������������� �
#__inference_bn1_layer_call_fn_16408�:;<=Q�N
G�D
:�7
inputs+��������������������������� 
p 

 
� ";�8
unknown+��������������������������� �
>__inference_bn2_layer_call_and_return_conditional_losses_16540�TUVWQ�N
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
>__inference_bn2_layer_call_and_return_conditional_losses_16558�TUVWQ�N
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
#__inference_bn2_layer_call_fn_16509�TUVWQ�N
G�D
:�7
inputs+���������������������������@
p

 
� ";�8
unknown+���������������������������@�
#__inference_bn2_layer_call_fn_16522�TUVWQ�N
G�D
:�7
inputs+���������������������������@
p 

 
� ";�8
unknown+���������������������������@�
>__inference_bn3_layer_call_and_return_conditional_losses_16654�nopqQ�N
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
>__inference_bn3_layer_call_and_return_conditional_losses_16672�nopqQ�N
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
#__inference_bn3_layer_call_fn_16623�nopqQ�N
G�D
:�7
inputs+���������������������������@
p

 
� ";�8
unknown+���������������������������@�
#__inference_bn3_layer_call_fn_16636�nopqQ�N
G�D
:�7
inputs+���������������������������@
p 

 
� ";�8
unknown+���������������������������@�
>__inference_bn4_layer_call_and_return_conditional_losses_16768�����Q�N
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
>__inference_bn4_layer_call_and_return_conditional_losses_16786�����Q�N
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
#__inference_bn4_layer_call_fn_16737�����Q�N
G�D
:�7
inputs+���������������������������@
p

 
� ";�8
unknown+���������������������������@�
#__inference_bn4_layer_call_fn_16750�����Q�N
G�D
:�7
inputs+���������������������������@
p 

 
� ";�8
unknown+���������������������������@�
>__inference_bn5_layer_call_and_return_conditional_losses_16882�����Q�N
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
>__inference_bn5_layer_call_and_return_conditional_losses_16900�����Q�N
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
#__inference_bn5_layer_call_fn_16851�����Q�N
G�D
:�7
inputs+��������������������������� 
p

 
� ";�8
unknown+��������������������������� �
#__inference_bn5_layer_call_fn_16864�����Q�N
G�D
:�7
inputs+��������������������������� 
p 

 
� ";�8
unknown+��������������������������� �
K__inference_convTranspose2D1_layer_call_and_return_conditional_losses_16382�01I�F
?�<
:�7
inputs+��������������������������� 
� "F�C
<�9
tensor_0+��������������������������� 
� �
0__inference_convTranspose2D1_layer_call_fn_16349�01I�F
?�<
:�7
inputs+��������������������������� 
� ";�8
unknown+��������������������������� �
K__inference_convTranspose2D2_layer_call_and_return_conditional_losses_16496�JKI�F
?�<
:�7
inputs+��������������������������� 
� "F�C
<�9
tensor_0+���������������������������@
� �
0__inference_convTranspose2D2_layer_call_fn_16463�JKI�F
?�<
:�7
inputs+��������������������������� 
� ";�8
unknown+���������������������������@�
K__inference_convTranspose2D3_layer_call_and_return_conditional_losses_16610�deI�F
?�<
:�7
inputs+���������������������������@
� "F�C
<�9
tensor_0+���������������������������@
� �
0__inference_convTranspose2D3_layer_call_fn_16577�deI�F
?�<
:�7
inputs+���������������������������@
� ";�8
unknown+���������������������������@�
K__inference_convTranspose2D4_layer_call_and_return_conditional_losses_16724�~I�F
?�<
:�7
inputs+���������������������������@
� "F�C
<�9
tensor_0+���������������������������@
� �
0__inference_convTranspose2D4_layer_call_fn_16691�~I�F
?�<
:�7
inputs+���������������������������@
� ";�8
unknown+���������������������������@�
K__inference_convTranspose2D5_layer_call_and_return_conditional_losses_16838���I�F
?�<
:�7
inputs+���������������������������@
� "F�C
<�9
tensor_0+��������������������������� 
� �
0__inference_convTranspose2D5_layer_call_fn_16805���I�F
?�<
:�7
inputs+���������������������������@
� ";�8
unknown+��������������������������� �
K__inference_convTranspose2D6_layer_call_and_return_conditional_losses_16953���I�F
?�<
:�7
inputs+��������������������������� 
� "F�C
<�9
tensor_0+���������������������������
� �
0__inference_convTranspose2D6_layer_call_fn_16919���I�F
?�<
:�7
inputs+��������������������������� 
� ";�8
unknown+����������������������������
B__inference_decoder_layer_call_and_return_conditional_losses_15052�."#01:;<=JKTUVWdenopq~������������;�8
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
B__inference_decoder_layer_call_and_return_conditional_losses_15142�."#01:;<=JKTUVWdenopq~������������;�8
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
B__inference_decoder_layer_call_and_return_conditional_losses_16092�."#01:;<=JKTUVWdenopq~������������7�4
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
B__inference_decoder_layer_call_and_return_conditional_losses_16302�."#01:;<=JKTUVWdenopq~������������7�4
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
'__inference_decoder_layer_call_fn_15306�."#01:;<=JKTUVWdenopq~������������;�8
1�.
$�!

inputLayer���������

p

 
� ")�&
unknown����������
'__inference_decoder_layer_call_fn_15469�."#01:;<=JKTUVWdenopq~������������;�8
1�.
$�!

inputLayer���������

p 

 
� ")�&
unknown����������
'__inference_decoder_layer_call_fn_15809�."#01:;<=JKTUVWdenopq~������������7�4
-�*
 �
inputs���������

p

 
� ")�&
unknown����������
'__inference_decoder_layer_call_fn_15882�."#01:;<=JKTUVWdenopq~������������7�4
-�*
 �
inputs���������

p 

 
� ")�&
unknown����������
A__inference_dense1_layer_call_and_return_conditional_losses_16321d"#/�,
%�"
 �
inputs���������

� "-�*
#� 
tensor_0����������
� �
&__inference_dense1_layer_call_fn_16311Y"#/�,
%�"
 �
inputs���������

� ""�
unknown�����������
A__inference_lReLU1_layer_call_and_return_conditional_losses_16454o7�4
-�*
(�%
inputs��������� 
� "4�1
*�'
tensor_0��������� 
� �
&__inference_lReLU1_layer_call_fn_16449d7�4
-�*
(�%
inputs��������� 
� ")�&
unknown��������� �
A__inference_lReLU2_layer_call_and_return_conditional_losses_16568o7�4
-�*
(�%
inputs���������@
� "4�1
*�'
tensor_0���������@
� �
&__inference_lReLU2_layer_call_fn_16563d7�4
-�*
(�%
inputs���������@
� ")�&
unknown���������@�
A__inference_lReLU3_layer_call_and_return_conditional_losses_16682o7�4
-�*
(�%
inputs���������@
� "4�1
*�'
tensor_0���������@
� �
&__inference_lReLU3_layer_call_fn_16677d7�4
-�*
(�%
inputs���������@
� ")�&
unknown���������@�
A__inference_lReLU4_layer_call_and_return_conditional_losses_16796o7�4
-�*
(�%
inputs���������@
� "4�1
*�'
tensor_0���������@
� �
&__inference_lReLU4_layer_call_fn_16791d7�4
-�*
(�%
inputs���������@
� ")�&
unknown���������@�
A__inference_lReLU5_layer_call_and_return_conditional_losses_16910o7�4
-�*
(�%
inputs��������� 
� "4�1
*�'
tensor_0��������� 
� �
&__inference_lReLU5_layer_call_fn_16905d7�4
-�*
(�%
inputs��������� 
� ")�&
unknown��������� �
G__inference_reshapeLayer_layer_call_and_return_conditional_losses_16340h0�-
&�#
!�
inputs����������
� "4�1
*�'
tensor_0��������� 
� �
,__inference_reshapeLayer_layer_call_fn_16326]0�-
&�#
!�
inputs����������
� ")�&
unknown��������� �
#__inference_signature_wrapper_15736�."#01:;<=JKTUVWdenopq~������������A�>
� 
7�4
2

inputLayer$�!

inputlayer���������
"K�H
F
convTranspose2D62�/
convtranspose2d6���������