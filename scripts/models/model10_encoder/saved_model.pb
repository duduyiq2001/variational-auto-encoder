��
��
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
Conv2D

input"T
filter"T
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
H
ShardedFilename
basename	
shard

num_shards
filename
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
 �"serve*2.12.02v2.12.0-rc1-12-g0db597d0d758��
n
logVar/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*
shared_namelogVar/bias
g
logVar/bias/Read/ReadVariableOpReadVariableOplogVar/bias*
_output_shapes
:
*
dtype0
w
logVar/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	�
*
shared_namelogVar/kernel
p
!logVar/kernel/Read/ReadVariableOpReadVariableOplogVar/kernel*
_output_shapes
:	�
*
dtype0
j
	mean/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*
shared_name	mean/bias
c
mean/bias/Read/ReadVariableOpReadVariableOp	mean/bias*
_output_shapes
:
*
dtype0
s
mean/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	�
*
shared_namemean/kernel
l
mean/kernel/Read/ReadVariableOpReadVariableOpmean/kernel*
_output_shapes
:	�
*
dtype0
~
bn6/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape: *$
shared_namebn6/moving_variance
w
'bn6/moving_variance/Read/ReadVariableOpReadVariableOpbn6/moving_variance*
_output_shapes
: *
dtype0
v
bn6/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape: * 
shared_namebn6/moving_mean
o
#bn6/moving_mean/Read/ReadVariableOpReadVariableOpbn6/moving_mean*
_output_shapes
: *
dtype0
h
bn6/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name
bn6/beta
a
bn6/beta/Read/ReadVariableOpReadVariableOpbn6/beta*
_output_shapes
: *
dtype0
j
	bn6/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	bn6/gamma
c
bn6/gamma/Read/ReadVariableOpReadVariableOp	bn6/gamma*
_output_shapes
: *
dtype0
p
conv2D6/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameconv2D6/bias
i
 conv2D6/bias/Read/ReadVariableOpReadVariableOpconv2D6/bias*
_output_shapes
: *
dtype0
�
conv2D6/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@ *
shared_nameconv2D6/kernel
y
"conv2D6/kernel/Read/ReadVariableOpReadVariableOpconv2D6/kernel*&
_output_shapes
:@ *
dtype0
~
bn5/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*$
shared_namebn5/moving_variance
w
'bn5/moving_variance/Read/ReadVariableOpReadVariableOpbn5/moving_variance*
_output_shapes
:@*
dtype0
v
bn5/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:@* 
shared_namebn5/moving_mean
o
#bn5/moving_mean/Read/ReadVariableOpReadVariableOpbn5/moving_mean*
_output_shapes
:@*
dtype0
h
bn5/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*
shared_name
bn5/beta
a
bn5/beta/Read/ReadVariableOpReadVariableOpbn5/beta*
_output_shapes
:@*
dtype0
j
	bn5/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*
shared_name	bn5/gamma
c
bn5/gamma/Read/ReadVariableOpReadVariableOp	bn5/gamma*
_output_shapes
:@*
dtype0
p
conv2D5/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*
shared_nameconv2D5/bias
i
 conv2D5/bias/Read/ReadVariableOpReadVariableOpconv2D5/bias*
_output_shapes
:@*
dtype0
�
conv2D5/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@@*
shared_nameconv2D5/kernel
y
"conv2D5/kernel/Read/ReadVariableOpReadVariableOpconv2D5/kernel*&
_output_shapes
:@@*
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
p
conv2D4/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*
shared_nameconv2D4/bias
i
 conv2D4/bias/Read/ReadVariableOpReadVariableOpconv2D4/bias*
_output_shapes
:@*
dtype0
�
conv2D4/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@@*
shared_nameconv2D4/kernel
y
"conv2D4/kernel/Read/ReadVariableOpReadVariableOpconv2D4/kernel*&
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
p
conv2D3/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*
shared_nameconv2D3/bias
i
 conv2D3/bias/Read/ReadVariableOpReadVariableOpconv2D3/bias*
_output_shapes
:@*
dtype0
�
conv2D3/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@@*
shared_nameconv2D3/kernel
y
"conv2D3/kernel/Read/ReadVariableOpReadVariableOpconv2D3/kernel*&
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
p
conv2D2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*
shared_nameconv2D2/bias
i
 conv2D2/bias/Read/ReadVariableOpReadVariableOpconv2D2/bias*
_output_shapes
:@*
dtype0
�
conv2D2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: @*
shared_nameconv2D2/kernel
y
"conv2D2/kernel/Read/ReadVariableOpReadVariableOpconv2D2/kernel*&
_output_shapes
: @*
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
p
conv2D1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameconv2D1/bias
i
 conv2D1/bias/Read/ReadVariableOpReadVariableOpconv2D1/bias*
_output_shapes
: *
dtype0
�
conv2D1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameconv2D1/kernel
y
"conv2D1/kernel/Read/ReadVariableOpReadVariableOpconv2D1/kernel*&
_output_shapes
: *
dtype0
�
serving_default_input_layerPlaceholder*/
_output_shapes
:���������*
dtype0*$
shape:���������
�
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_layerconv2D1/kernelconv2D1/bias	bn1/gammabn1/betabn1/moving_meanbn1/moving_varianceconv2D2/kernelconv2D2/bias	bn2/gammabn2/betabn2/moving_meanbn2/moving_varianceconv2D3/kernelconv2D3/bias	bn3/gammabn3/betabn3/moving_meanbn3/moving_varianceconv2D4/kernelconv2D4/bias	bn4/gammabn4/betabn4/moving_meanbn4/moving_varianceconv2D5/kernelconv2D5/bias	bn5/gammabn5/betabn5/moving_meanbn5/moving_varianceconv2D6/kernelconv2D6/bias	bn6/gammabn6/betabn6/moving_meanbn6/moving_variancelogVar/kernellogVar/biasmean/kernel	mean/bias*4
Tin-
+2)*
Tout
2*
_collective_manager_ids
 *:
_output_shapes(
&:���������
:���������
*J
_read_only_resource_inputs,
*(	
 !"#$%&'(*-
config_proto

CPU

GPU 2J 8� *,
f'R%
#__inference_signature_wrapper_12481

NoOpNoOp
��
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*�
value�Bސ B֐
�
layer-0
layer_with_weights-0
layer-1
layer_with_weights-1
layer-2
layer-3
layer_with_weights-2
layer-4
layer_with_weights-3
layer-5
layer-6
layer_with_weights-4
layer-7
	layer_with_weights-5
	layer-8

layer-9
layer_with_weights-6
layer-10
layer_with_weights-7
layer-11
layer-12
layer_with_weights-8
layer-13
layer_with_weights-9
layer-14
layer-15
layer_with_weights-10
layer-16
layer_with_weights-11
layer-17
layer-18
layer-19
layer_with_weights-12
layer-20
layer_with_weights-13
layer-21
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
_default_save_signature

signatures*
* 
�
	variables
 trainable_variables
!regularization_losses
"	keras_api
#__call__
*$&call_and_return_all_conditional_losses

%kernel
&bias
 '_jit_compiled_convolution_op*
�
(	variables
)trainable_variables
*regularization_losses
+	keras_api
,__call__
*-&call_and_return_all_conditional_losses
.axis
	/gamma
0beta
1moving_mean
2moving_variance*
�
3	variables
4trainable_variables
5regularization_losses
6	keras_api
7__call__
*8&call_and_return_all_conditional_losses* 
�
9	variables
:trainable_variables
;regularization_losses
<	keras_api
=__call__
*>&call_and_return_all_conditional_losses

?kernel
@bias
 A_jit_compiled_convolution_op*
�
B	variables
Ctrainable_variables
Dregularization_losses
E	keras_api
F__call__
*G&call_and_return_all_conditional_losses
Haxis
	Igamma
Jbeta
Kmoving_mean
Lmoving_variance*
�
M	variables
Ntrainable_variables
Oregularization_losses
P	keras_api
Q__call__
*R&call_and_return_all_conditional_losses* 
�
S	variables
Ttrainable_variables
Uregularization_losses
V	keras_api
W__call__
*X&call_and_return_all_conditional_losses

Ykernel
Zbias
 [_jit_compiled_convolution_op*
�
\	variables
]trainable_variables
^regularization_losses
_	keras_api
`__call__
*a&call_and_return_all_conditional_losses
baxis
	cgamma
dbeta
emoving_mean
fmoving_variance*
�
g	variables
htrainable_variables
iregularization_losses
j	keras_api
k__call__
*l&call_and_return_all_conditional_losses* 
�
m	variables
ntrainable_variables
oregularization_losses
p	keras_api
q__call__
*r&call_and_return_all_conditional_losses

skernel
tbias
 u_jit_compiled_convolution_op*
�
v	variables
wtrainable_variables
xregularization_losses
y	keras_api
z__call__
*{&call_and_return_all_conditional_losses
|axis
	}gamma
~beta
moving_mean
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
+�&call_and_return_all_conditional_losses* 
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�kernel
	�bias*
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�kernel
	�bias*
�
%0
&1
/2
03
14
25
?6
@7
I8
J9
K10
L11
Y12
Z13
c14
d15
e16
f17
s18
t19
}20
~21
22
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
�33
�34
�35
�36
�37
�38
�39*
�
%0
&1
/2
03
?4
@5
I6
J7
Y8
Z9
c10
d11
s12
t13
}14
~15
�16
�17
�18
�19
�20
�21
�22
�23
�24
�25
�26
�27*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
	variables
trainable_variables
regularization_losses
__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*
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
%0
&1*

%0
&1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
	variables
 trainable_variables
!regularization_losses
#__call__
*$&call_and_return_all_conditional_losses
&$"call_and_return_conditional_losses*

�trace_0* 

�trace_0* 
^X
VARIABLE_VALUEconv2D1/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE*
ZT
VARIABLE_VALUEconv2D1/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 
 
/0
01
12
23*

/0
01*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
(	variables
)trainable_variables
*regularization_losses
,__call__
*-&call_and_return_all_conditional_losses
&-"call_and_return_conditional_losses*

�trace_0
�trace_1* 

�trace_0
�trace_1* 
* 
XR
VARIABLE_VALUE	bn1/gamma5layer_with_weights-1/gamma/.ATTRIBUTES/VARIABLE_VALUE*
VP
VARIABLE_VALUEbn1/beta4layer_with_weights-1/beta/.ATTRIBUTES/VARIABLE_VALUE*
d^
VARIABLE_VALUEbn1/moving_mean;layer_with_weights-1/moving_mean/.ATTRIBUTES/VARIABLE_VALUE*
lf
VARIABLE_VALUEbn1/moving_variance?layer_with_weights-1/moving_variance/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
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
&8"call_and_return_conditional_losses* 

�trace_0* 

�trace_0* 

?0
@1*

?0
@1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
9	variables
:trainable_variables
;regularization_losses
=__call__
*>&call_and_return_all_conditional_losses
&>"call_and_return_conditional_losses*

�trace_0* 

�trace_0* 
^X
VARIABLE_VALUEconv2D2/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE*
ZT
VARIABLE_VALUEconv2D2/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 
 
I0
J1
K2
L3*

I0
J1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
B	variables
Ctrainable_variables
Dregularization_losses
F__call__
*G&call_and_return_all_conditional_losses
&G"call_and_return_conditional_losses*

�trace_0
�trace_1* 

�trace_0
�trace_1* 
* 
XR
VARIABLE_VALUE	bn2/gamma5layer_with_weights-3/gamma/.ATTRIBUTES/VARIABLE_VALUE*
VP
VARIABLE_VALUEbn2/beta4layer_with_weights-3/beta/.ATTRIBUTES/VARIABLE_VALUE*
d^
VARIABLE_VALUEbn2/moving_mean;layer_with_weights-3/moving_mean/.ATTRIBUTES/VARIABLE_VALUE*
lf
VARIABLE_VALUEbn2/moving_variance?layer_with_weights-3/moving_variance/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
M	variables
Ntrainable_variables
Oregularization_losses
Q__call__
*R&call_and_return_all_conditional_losses
&R"call_and_return_conditional_losses* 

�trace_0* 

�trace_0* 

Y0
Z1*

Y0
Z1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
S	variables
Ttrainable_variables
Uregularization_losses
W__call__
*X&call_and_return_all_conditional_losses
&X"call_and_return_conditional_losses*

�trace_0* 

�trace_0* 
^X
VARIABLE_VALUEconv2D3/kernel6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUE*
ZT
VARIABLE_VALUEconv2D3/bias4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 
 
c0
d1
e2
f3*

c0
d1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
\	variables
]trainable_variables
^regularization_losses
`__call__
*a&call_and_return_all_conditional_losses
&a"call_and_return_conditional_losses*

�trace_0
�trace_1* 

�trace_0
�trace_1* 
* 
XR
VARIABLE_VALUE	bn3/gamma5layer_with_weights-5/gamma/.ATTRIBUTES/VARIABLE_VALUE*
VP
VARIABLE_VALUEbn3/beta4layer_with_weights-5/beta/.ATTRIBUTES/VARIABLE_VALUE*
d^
VARIABLE_VALUEbn3/moving_mean;layer_with_weights-5/moving_mean/.ATTRIBUTES/VARIABLE_VALUE*
lf
VARIABLE_VALUEbn3/moving_variance?layer_with_weights-5/moving_variance/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
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
&l"call_and_return_conditional_losses* 

�trace_0* 

�trace_0* 

s0
t1*

s0
t1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
m	variables
ntrainable_variables
oregularization_losses
q__call__
*r&call_and_return_all_conditional_losses
&r"call_and_return_conditional_losses*

�trace_0* 

�trace_0* 
^X
VARIABLE_VALUEconv2D4/kernel6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUE*
ZT
VARIABLE_VALUEconv2D4/bias4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 
!
}0
~1
2
�3*

}0
~1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
v	variables
wtrainable_variables
xregularization_losses
z__call__
*{&call_and_return_all_conditional_losses
&{"call_and_return_conditional_losses*

�trace_0
�trace_1* 

�trace_0
�trace_1* 
* 
XR
VARIABLE_VALUE	bn4/gamma5layer_with_weights-7/gamma/.ATTRIBUTES/VARIABLE_VALUE*
VP
VARIABLE_VALUEbn4/beta4layer_with_weights-7/beta/.ATTRIBUTES/VARIABLE_VALUE*
d^
VARIABLE_VALUEbn4/moving_mean;layer_with_weights-7/moving_mean/.ATTRIBUTES/VARIABLE_VALUE*
lf
VARIABLE_VALUEbn4/moving_variance?layer_with_weights-7/moving_variance/.ATTRIBUTES/VARIABLE_VALUE*
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
^X
VARIABLE_VALUEconv2D5/kernel6layer_with_weights-8/kernel/.ATTRIBUTES/VARIABLE_VALUE*
ZT
VARIABLE_VALUEconv2D5/bias4layer_with_weights-8/bias/.ATTRIBUTES/VARIABLE_VALUE*
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
VARIABLE_VALUE	bn5/gamma5layer_with_weights-9/gamma/.ATTRIBUTES/VARIABLE_VALUE*
VP
VARIABLE_VALUEbn5/beta4layer_with_weights-9/beta/.ATTRIBUTES/VARIABLE_VALUE*
d^
VARIABLE_VALUEbn5/moving_mean;layer_with_weights-9/moving_mean/.ATTRIBUTES/VARIABLE_VALUE*
lf
VARIABLE_VALUEbn5/moving_variance?layer_with_weights-9/moving_variance/.ATTRIBUTES/VARIABLE_VALUE*
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
_Y
VARIABLE_VALUEconv2D6/kernel7layer_with_weights-10/kernel/.ATTRIBUTES/VARIABLE_VALUE*
[U
VARIABLE_VALUEconv2D6/bias5layer_with_weights-10/bias/.ATTRIBUTES/VARIABLE_VALUE*
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
VARIABLE_VALUE	bn6/gamma6layer_with_weights-11/gamma/.ATTRIBUTES/VARIABLE_VALUE*
WQ
VARIABLE_VALUEbn6/beta5layer_with_weights-11/beta/.ATTRIBUTES/VARIABLE_VALUE*
e_
VARIABLE_VALUEbn6/moving_mean<layer_with_weights-11/moving_mean/.ATTRIBUTES/VARIABLE_VALUE*
mg
VARIABLE_VALUEbn6/moving_variance@layer_with_weights-11/moving_variance/.ATTRIBUTES/VARIABLE_VALUE*
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
\V
VARIABLE_VALUEmean/kernel7layer_with_weights-12/kernel/.ATTRIBUTES/VARIABLE_VALUE*
XR
VARIABLE_VALUE	mean/bias5layer_with_weights-12/bias/.ATTRIBUTES/VARIABLE_VALUE*
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
^X
VARIABLE_VALUElogVar/kernel7layer_with_weights-13/kernel/.ATTRIBUTES/VARIABLE_VALUE*
ZT
VARIABLE_VALUElogVar/bias5layer_with_weights-13/bias/.ATTRIBUTES/VARIABLE_VALUE*
_
10
21
K2
L3
e4
f5
6
�7
�8
�9
�10
�11*
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
18
19
20
21*
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
10
21*
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
K0
L1*
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
e0
f1*
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

0
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
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filenameconv2D1/kernelconv2D1/bias	bn1/gammabn1/betabn1/moving_meanbn1/moving_varianceconv2D2/kernelconv2D2/bias	bn2/gammabn2/betabn2/moving_meanbn2/moving_varianceconv2D3/kernelconv2D3/bias	bn3/gammabn3/betabn3/moving_meanbn3/moving_varianceconv2D4/kernelconv2D4/bias	bn4/gammabn4/betabn4/moving_meanbn4/moving_varianceconv2D5/kernelconv2D5/bias	bn5/gammabn5/betabn5/moving_meanbn5/moving_varianceconv2D6/kernelconv2D6/bias	bn6/gammabn6/betabn6/moving_meanbn6/moving_variancemean/kernel	mean/biaslogVar/kernellogVar/biasConst*5
Tin.
,2**
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
__inference__traced_save_13804
�
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenameconv2D1/kernelconv2D1/bias	bn1/gammabn1/betabn1/moving_meanbn1/moving_varianceconv2D2/kernelconv2D2/bias	bn2/gammabn2/betabn2/moving_meanbn2/moving_varianceconv2D3/kernelconv2D3/bias	bn3/gammabn3/betabn3/moving_meanbn3/moving_varianceconv2D4/kernelconv2D4/bias	bn4/gammabn4/betabn4/moving_meanbn4/moving_varianceconv2D5/kernelconv2D5/bias	bn5/gammabn5/betabn5/moving_meanbn5/moving_varianceconv2D6/kernelconv2D6/bias	bn6/gammabn6/betabn6/moving_meanbn6/moving_variancemean/kernel	mean/biaslogVar/kernellogVar/bias*4
Tin-
+2)*
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
!__inference__traced_restore_13934��
�
]
A__inference_lReLU2_layer_call_and_return_conditional_losses_11431

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
�
]
A__inference_lReLU6_layer_call_and_return_conditional_losses_13491

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
�
C
'__inference_flatten_layer_call_fn_13496

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *K
fFRD
B__inference_flatten_layer_call_and_return_conditional_losses_11567a
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:����������"
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
>__inference_bn5_layer_call_and_return_conditional_losses_13372

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
'__inference_conv2D5_layer_call_fn_13318

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
 */
_output_shapes
:���������@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *K
fFRD
B__inference_conv2D5_layer_call_and_return_conditional_losses_11507w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:���������@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������@: : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:���������@
 
_user_specified_nameinputs
�
�
>__inference_bn3_layer_call_and_return_conditional_losses_13208

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
�
^
B__inference_flatten_layer_call_and_return_conditional_losses_11567

inputs
identityV
ConstConst*
_output_shapes
:*
dtype0*
valueB"����   ]
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:����������Y
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:����������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:��������� :W S
/
_output_shapes
:��������� 
 
_user_specified_nameinputs
�
]
A__inference_lReLU1_layer_call_and_return_conditional_losses_11399

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
��
�"
__inference__traced_save_13804
file_prefix?
%read_disablecopyonread_conv2d1_kernel: 3
%read_1_disablecopyonread_conv2d1_bias: 0
"read_2_disablecopyonread_bn1_gamma: /
!read_3_disablecopyonread_bn1_beta: 6
(read_4_disablecopyonread_bn1_moving_mean: :
,read_5_disablecopyonread_bn1_moving_variance: A
'read_6_disablecopyonread_conv2d2_kernel: @3
%read_7_disablecopyonread_conv2d2_bias:@0
"read_8_disablecopyonread_bn2_gamma:@/
!read_9_disablecopyonread_bn2_beta:@7
)read_10_disablecopyonread_bn2_moving_mean:@;
-read_11_disablecopyonread_bn2_moving_variance:@B
(read_12_disablecopyonread_conv2d3_kernel:@@4
&read_13_disablecopyonread_conv2d3_bias:@1
#read_14_disablecopyonread_bn3_gamma:@0
"read_15_disablecopyonread_bn3_beta:@7
)read_16_disablecopyonread_bn3_moving_mean:@;
-read_17_disablecopyonread_bn3_moving_variance:@B
(read_18_disablecopyonread_conv2d4_kernel:@@4
&read_19_disablecopyonread_conv2d4_bias:@1
#read_20_disablecopyonread_bn4_gamma:@0
"read_21_disablecopyonread_bn4_beta:@7
)read_22_disablecopyonread_bn4_moving_mean:@;
-read_23_disablecopyonread_bn4_moving_variance:@B
(read_24_disablecopyonread_conv2d5_kernel:@@4
&read_25_disablecopyonread_conv2d5_bias:@1
#read_26_disablecopyonread_bn5_gamma:@0
"read_27_disablecopyonread_bn5_beta:@7
)read_28_disablecopyonread_bn5_moving_mean:@;
-read_29_disablecopyonread_bn5_moving_variance:@B
(read_30_disablecopyonread_conv2d6_kernel:@ 4
&read_31_disablecopyonread_conv2d6_bias: 1
#read_32_disablecopyonread_bn6_gamma: 0
"read_33_disablecopyonread_bn6_beta: 7
)read_34_disablecopyonread_bn6_moving_mean: ;
-read_35_disablecopyonread_bn6_moving_variance: 8
%read_36_disablecopyonread_mean_kernel:	�
1
#read_37_disablecopyonread_mean_bias:
:
'read_38_disablecopyonread_logvar_kernel:	�
3
%read_39_disablecopyonread_logvar_bias:

savev2_const
identity_81��MergeV2Checkpoints�Read/DisableCopyOnRead�Read/ReadVariableOp�Read_1/DisableCopyOnRead�Read_1/ReadVariableOp�Read_10/DisableCopyOnRead�Read_10/ReadVariableOp�Read_11/DisableCopyOnRead�Read_11/ReadVariableOp�Read_12/DisableCopyOnRead�Read_12/ReadVariableOp�Read_13/DisableCopyOnRead�Read_13/ReadVariableOp�Read_14/DisableCopyOnRead�Read_14/ReadVariableOp�Read_15/DisableCopyOnRead�Read_15/ReadVariableOp�Read_16/DisableCopyOnRead�Read_16/ReadVariableOp�Read_17/DisableCopyOnRead�Read_17/ReadVariableOp�Read_18/DisableCopyOnRead�Read_18/ReadVariableOp�Read_19/DisableCopyOnRead�Read_19/ReadVariableOp�Read_2/DisableCopyOnRead�Read_2/ReadVariableOp�Read_20/DisableCopyOnRead�Read_20/ReadVariableOp�Read_21/DisableCopyOnRead�Read_21/ReadVariableOp�Read_22/DisableCopyOnRead�Read_22/ReadVariableOp�Read_23/DisableCopyOnRead�Read_23/ReadVariableOp�Read_24/DisableCopyOnRead�Read_24/ReadVariableOp�Read_25/DisableCopyOnRead�Read_25/ReadVariableOp�Read_26/DisableCopyOnRead�Read_26/ReadVariableOp�Read_27/DisableCopyOnRead�Read_27/ReadVariableOp�Read_28/DisableCopyOnRead�Read_28/ReadVariableOp�Read_29/DisableCopyOnRead�Read_29/ReadVariableOp�Read_3/DisableCopyOnRead�Read_3/ReadVariableOp�Read_30/DisableCopyOnRead�Read_30/ReadVariableOp�Read_31/DisableCopyOnRead�Read_31/ReadVariableOp�Read_32/DisableCopyOnRead�Read_32/ReadVariableOp�Read_33/DisableCopyOnRead�Read_33/ReadVariableOp�Read_34/DisableCopyOnRead�Read_34/ReadVariableOp�Read_35/DisableCopyOnRead�Read_35/ReadVariableOp�Read_36/DisableCopyOnRead�Read_36/ReadVariableOp�Read_37/DisableCopyOnRead�Read_37/ReadVariableOp�Read_38/DisableCopyOnRead�Read_38/ReadVariableOp�Read_39/DisableCopyOnRead�Read_39/ReadVariableOp�Read_4/DisableCopyOnRead�Read_4/ReadVariableOp�Read_5/DisableCopyOnRead�Read_5/ReadVariableOp�Read_6/DisableCopyOnRead�Read_6/ReadVariableOp�Read_7/DisableCopyOnRead�Read_7/ReadVariableOp�Read_8/DisableCopyOnRead�Read_8/ReadVariableOp�Read_9/DisableCopyOnRead�Read_9/ReadVariableOpw
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
: w
Read/DisableCopyOnReadDisableCopyOnRead%read_disablecopyonread_conv2d1_kernel"/device:CPU:0*
_output_shapes
 �
Read/ReadVariableOpReadVariableOp%read_disablecopyonread_conv2d1_kernel^Read/DisableCopyOnRead"/device:CPU:0*&
_output_shapes
: *
dtype0q
IdentityIdentityRead/ReadVariableOp:value:0"/device:CPU:0*
T0*&
_output_shapes
: i

Identity_1IdentityIdentity:output:0"/device:CPU:0*
T0*&
_output_shapes
: y
Read_1/DisableCopyOnReadDisableCopyOnRead%read_1_disablecopyonread_conv2d1_bias"/device:CPU:0*
_output_shapes
 �
Read_1/ReadVariableOpReadVariableOp%read_1_disablecopyonread_conv2d1_bias^Read_1/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0i

Identity_2IdentityRead_1/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: _

Identity_3IdentityIdentity_2:output:0"/device:CPU:0*
T0*
_output_shapes
: v
Read_2/DisableCopyOnReadDisableCopyOnRead"read_2_disablecopyonread_bn1_gamma"/device:CPU:0*
_output_shapes
 �
Read_2/ReadVariableOpReadVariableOp"read_2_disablecopyonread_bn1_gamma^Read_2/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0i

Identity_4IdentityRead_2/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: _

Identity_5IdentityIdentity_4:output:0"/device:CPU:0*
T0*
_output_shapes
: u
Read_3/DisableCopyOnReadDisableCopyOnRead!read_3_disablecopyonread_bn1_beta"/device:CPU:0*
_output_shapes
 �
Read_3/ReadVariableOpReadVariableOp!read_3_disablecopyonread_bn1_beta^Read_3/DisableCopyOnRead"/device:CPU:0*
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
: |
Read_4/DisableCopyOnReadDisableCopyOnRead(read_4_disablecopyonread_bn1_moving_mean"/device:CPU:0*
_output_shapes
 �
Read_4/ReadVariableOpReadVariableOp(read_4_disablecopyonread_bn1_moving_mean^Read_4/DisableCopyOnRead"/device:CPU:0*
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
: �
Read_5/DisableCopyOnReadDisableCopyOnRead,read_5_disablecopyonread_bn1_moving_variance"/device:CPU:0*
_output_shapes
 �
Read_5/ReadVariableOpReadVariableOp,read_5_disablecopyonread_bn1_moving_variance^Read_5/DisableCopyOnRead"/device:CPU:0*
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
: {
Read_6/DisableCopyOnReadDisableCopyOnRead'read_6_disablecopyonread_conv2d2_kernel"/device:CPU:0*
_output_shapes
 �
Read_6/ReadVariableOpReadVariableOp'read_6_disablecopyonread_conv2d2_kernel^Read_6/DisableCopyOnRead"/device:CPU:0*&
_output_shapes
: @*
dtype0v
Identity_12IdentityRead_6/ReadVariableOp:value:0"/device:CPU:0*
T0*&
_output_shapes
: @m
Identity_13IdentityIdentity_12:output:0"/device:CPU:0*
T0*&
_output_shapes
: @y
Read_7/DisableCopyOnReadDisableCopyOnRead%read_7_disablecopyonread_conv2d2_bias"/device:CPU:0*
_output_shapes
 �
Read_7/ReadVariableOpReadVariableOp%read_7_disablecopyonread_conv2d2_bias^Read_7/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:@*
dtype0j
Identity_14IdentityRead_7/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:@a
Identity_15IdentityIdentity_14:output:0"/device:CPU:0*
T0*
_output_shapes
:@v
Read_8/DisableCopyOnReadDisableCopyOnRead"read_8_disablecopyonread_bn2_gamma"/device:CPU:0*
_output_shapes
 �
Read_8/ReadVariableOpReadVariableOp"read_8_disablecopyonread_bn2_gamma^Read_8/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:@*
dtype0j
Identity_16IdentityRead_8/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:@a
Identity_17IdentityIdentity_16:output:0"/device:CPU:0*
T0*
_output_shapes
:@u
Read_9/DisableCopyOnReadDisableCopyOnRead!read_9_disablecopyonread_bn2_beta"/device:CPU:0*
_output_shapes
 �
Read_9/ReadVariableOpReadVariableOp!read_9_disablecopyonread_bn2_beta^Read_9/DisableCopyOnRead"/device:CPU:0*
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
:@~
Read_10/DisableCopyOnReadDisableCopyOnRead)read_10_disablecopyonread_bn2_moving_mean"/device:CPU:0*
_output_shapes
 �
Read_10/ReadVariableOpReadVariableOp)read_10_disablecopyonread_bn2_moving_mean^Read_10/DisableCopyOnRead"/device:CPU:0*
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
:@�
Read_11/DisableCopyOnReadDisableCopyOnRead-read_11_disablecopyonread_bn2_moving_variance"/device:CPU:0*
_output_shapes
 �
Read_11/ReadVariableOpReadVariableOp-read_11_disablecopyonread_bn2_moving_variance^Read_11/DisableCopyOnRead"/device:CPU:0*
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
:@}
Read_12/DisableCopyOnReadDisableCopyOnRead(read_12_disablecopyonread_conv2d3_kernel"/device:CPU:0*
_output_shapes
 �
Read_12/ReadVariableOpReadVariableOp(read_12_disablecopyonread_conv2d3_kernel^Read_12/DisableCopyOnRead"/device:CPU:0*&
_output_shapes
:@@*
dtype0w
Identity_24IdentityRead_12/ReadVariableOp:value:0"/device:CPU:0*
T0*&
_output_shapes
:@@m
Identity_25IdentityIdentity_24:output:0"/device:CPU:0*
T0*&
_output_shapes
:@@{
Read_13/DisableCopyOnReadDisableCopyOnRead&read_13_disablecopyonread_conv2d3_bias"/device:CPU:0*
_output_shapes
 �
Read_13/ReadVariableOpReadVariableOp&read_13_disablecopyonread_conv2d3_bias^Read_13/DisableCopyOnRead"/device:CPU:0*
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
:@x
Read_14/DisableCopyOnReadDisableCopyOnRead#read_14_disablecopyonread_bn3_gamma"/device:CPU:0*
_output_shapes
 �
Read_14/ReadVariableOpReadVariableOp#read_14_disablecopyonread_bn3_gamma^Read_14/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:@*
dtype0k
Identity_28IdentityRead_14/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:@a
Identity_29IdentityIdentity_28:output:0"/device:CPU:0*
T0*
_output_shapes
:@w
Read_15/DisableCopyOnReadDisableCopyOnRead"read_15_disablecopyonread_bn3_beta"/device:CPU:0*
_output_shapes
 �
Read_15/ReadVariableOpReadVariableOp"read_15_disablecopyonread_bn3_beta^Read_15/DisableCopyOnRead"/device:CPU:0*
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
:@~
Read_16/DisableCopyOnReadDisableCopyOnRead)read_16_disablecopyonread_bn3_moving_mean"/device:CPU:0*
_output_shapes
 �
Read_16/ReadVariableOpReadVariableOp)read_16_disablecopyonread_bn3_moving_mean^Read_16/DisableCopyOnRead"/device:CPU:0*
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
:@�
Read_17/DisableCopyOnReadDisableCopyOnRead-read_17_disablecopyonread_bn3_moving_variance"/device:CPU:0*
_output_shapes
 �
Read_17/ReadVariableOpReadVariableOp-read_17_disablecopyonread_bn3_moving_variance^Read_17/DisableCopyOnRead"/device:CPU:0*
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
:@}
Read_18/DisableCopyOnReadDisableCopyOnRead(read_18_disablecopyonread_conv2d4_kernel"/device:CPU:0*
_output_shapes
 �
Read_18/ReadVariableOpReadVariableOp(read_18_disablecopyonread_conv2d4_kernel^Read_18/DisableCopyOnRead"/device:CPU:0*&
_output_shapes
:@@*
dtype0w
Identity_36IdentityRead_18/ReadVariableOp:value:0"/device:CPU:0*
T0*&
_output_shapes
:@@m
Identity_37IdentityIdentity_36:output:0"/device:CPU:0*
T0*&
_output_shapes
:@@{
Read_19/DisableCopyOnReadDisableCopyOnRead&read_19_disablecopyonread_conv2d4_bias"/device:CPU:0*
_output_shapes
 �
Read_19/ReadVariableOpReadVariableOp&read_19_disablecopyonread_conv2d4_bias^Read_19/DisableCopyOnRead"/device:CPU:0*
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
:@x
Read_20/DisableCopyOnReadDisableCopyOnRead#read_20_disablecopyonread_bn4_gamma"/device:CPU:0*
_output_shapes
 �
Read_20/ReadVariableOpReadVariableOp#read_20_disablecopyonread_bn4_gamma^Read_20/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:@*
dtype0k
Identity_40IdentityRead_20/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:@a
Identity_41IdentityIdentity_40:output:0"/device:CPU:0*
T0*
_output_shapes
:@w
Read_21/DisableCopyOnReadDisableCopyOnRead"read_21_disablecopyonread_bn4_beta"/device:CPU:0*
_output_shapes
 �
Read_21/ReadVariableOpReadVariableOp"read_21_disablecopyonread_bn4_beta^Read_21/DisableCopyOnRead"/device:CPU:0*
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
:@~
Read_22/DisableCopyOnReadDisableCopyOnRead)read_22_disablecopyonread_bn4_moving_mean"/device:CPU:0*
_output_shapes
 �
Read_22/ReadVariableOpReadVariableOp)read_22_disablecopyonread_bn4_moving_mean^Read_22/DisableCopyOnRead"/device:CPU:0*
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
:@�
Read_23/DisableCopyOnReadDisableCopyOnRead-read_23_disablecopyonread_bn4_moving_variance"/device:CPU:0*
_output_shapes
 �
Read_23/ReadVariableOpReadVariableOp-read_23_disablecopyonread_bn4_moving_variance^Read_23/DisableCopyOnRead"/device:CPU:0*
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
:@}
Read_24/DisableCopyOnReadDisableCopyOnRead(read_24_disablecopyonread_conv2d5_kernel"/device:CPU:0*
_output_shapes
 �
Read_24/ReadVariableOpReadVariableOp(read_24_disablecopyonread_conv2d5_kernel^Read_24/DisableCopyOnRead"/device:CPU:0*&
_output_shapes
:@@*
dtype0w
Identity_48IdentityRead_24/ReadVariableOp:value:0"/device:CPU:0*
T0*&
_output_shapes
:@@m
Identity_49IdentityIdentity_48:output:0"/device:CPU:0*
T0*&
_output_shapes
:@@{
Read_25/DisableCopyOnReadDisableCopyOnRead&read_25_disablecopyonread_conv2d5_bias"/device:CPU:0*
_output_shapes
 �
Read_25/ReadVariableOpReadVariableOp&read_25_disablecopyonread_conv2d5_bias^Read_25/DisableCopyOnRead"/device:CPU:0*
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
:@x
Read_26/DisableCopyOnReadDisableCopyOnRead#read_26_disablecopyonread_bn5_gamma"/device:CPU:0*
_output_shapes
 �
Read_26/ReadVariableOpReadVariableOp#read_26_disablecopyonread_bn5_gamma^Read_26/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:@*
dtype0k
Identity_52IdentityRead_26/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:@a
Identity_53IdentityIdentity_52:output:0"/device:CPU:0*
T0*
_output_shapes
:@w
Read_27/DisableCopyOnReadDisableCopyOnRead"read_27_disablecopyonread_bn5_beta"/device:CPU:0*
_output_shapes
 �
Read_27/ReadVariableOpReadVariableOp"read_27_disablecopyonread_bn5_beta^Read_27/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:@*
dtype0k
Identity_54IdentityRead_27/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:@a
Identity_55IdentityIdentity_54:output:0"/device:CPU:0*
T0*
_output_shapes
:@~
Read_28/DisableCopyOnReadDisableCopyOnRead)read_28_disablecopyonread_bn5_moving_mean"/device:CPU:0*
_output_shapes
 �
Read_28/ReadVariableOpReadVariableOp)read_28_disablecopyonread_bn5_moving_mean^Read_28/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:@*
dtype0k
Identity_56IdentityRead_28/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:@a
Identity_57IdentityIdentity_56:output:0"/device:CPU:0*
T0*
_output_shapes
:@�
Read_29/DisableCopyOnReadDisableCopyOnRead-read_29_disablecopyonread_bn5_moving_variance"/device:CPU:0*
_output_shapes
 �
Read_29/ReadVariableOpReadVariableOp-read_29_disablecopyonread_bn5_moving_variance^Read_29/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:@*
dtype0k
Identity_58IdentityRead_29/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:@a
Identity_59IdentityIdentity_58:output:0"/device:CPU:0*
T0*
_output_shapes
:@}
Read_30/DisableCopyOnReadDisableCopyOnRead(read_30_disablecopyonread_conv2d6_kernel"/device:CPU:0*
_output_shapes
 �
Read_30/ReadVariableOpReadVariableOp(read_30_disablecopyonread_conv2d6_kernel^Read_30/DisableCopyOnRead"/device:CPU:0*&
_output_shapes
:@ *
dtype0w
Identity_60IdentityRead_30/ReadVariableOp:value:0"/device:CPU:0*
T0*&
_output_shapes
:@ m
Identity_61IdentityIdentity_60:output:0"/device:CPU:0*
T0*&
_output_shapes
:@ {
Read_31/DisableCopyOnReadDisableCopyOnRead&read_31_disablecopyonread_conv2d6_bias"/device:CPU:0*
_output_shapes
 �
Read_31/ReadVariableOpReadVariableOp&read_31_disablecopyonread_conv2d6_bias^Read_31/DisableCopyOnRead"/device:CPU:0*
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
: x
Read_32/DisableCopyOnReadDisableCopyOnRead#read_32_disablecopyonread_bn6_gamma"/device:CPU:0*
_output_shapes
 �
Read_32/ReadVariableOpReadVariableOp#read_32_disablecopyonread_bn6_gamma^Read_32/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0k
Identity_64IdentityRead_32/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: a
Identity_65IdentityIdentity_64:output:0"/device:CPU:0*
T0*
_output_shapes
: w
Read_33/DisableCopyOnReadDisableCopyOnRead"read_33_disablecopyonread_bn6_beta"/device:CPU:0*
_output_shapes
 �
Read_33/ReadVariableOpReadVariableOp"read_33_disablecopyonread_bn6_beta^Read_33/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0k
Identity_66IdentityRead_33/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: a
Identity_67IdentityIdentity_66:output:0"/device:CPU:0*
T0*
_output_shapes
: ~
Read_34/DisableCopyOnReadDisableCopyOnRead)read_34_disablecopyonread_bn6_moving_mean"/device:CPU:0*
_output_shapes
 �
Read_34/ReadVariableOpReadVariableOp)read_34_disablecopyonread_bn6_moving_mean^Read_34/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0k
Identity_68IdentityRead_34/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: a
Identity_69IdentityIdentity_68:output:0"/device:CPU:0*
T0*
_output_shapes
: �
Read_35/DisableCopyOnReadDisableCopyOnRead-read_35_disablecopyonread_bn6_moving_variance"/device:CPU:0*
_output_shapes
 �
Read_35/ReadVariableOpReadVariableOp-read_35_disablecopyonread_bn6_moving_variance^Read_35/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0k
Identity_70IdentityRead_35/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: a
Identity_71IdentityIdentity_70:output:0"/device:CPU:0*
T0*
_output_shapes
: z
Read_36/DisableCopyOnReadDisableCopyOnRead%read_36_disablecopyonread_mean_kernel"/device:CPU:0*
_output_shapes
 �
Read_36/ReadVariableOpReadVariableOp%read_36_disablecopyonread_mean_kernel^Read_36/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:	�
*
dtype0p
Identity_72IdentityRead_36/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:	�
f
Identity_73IdentityIdentity_72:output:0"/device:CPU:0*
T0*
_output_shapes
:	�
x
Read_37/DisableCopyOnReadDisableCopyOnRead#read_37_disablecopyonread_mean_bias"/device:CPU:0*
_output_shapes
 �
Read_37/ReadVariableOpReadVariableOp#read_37_disablecopyonread_mean_bias^Read_37/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:
*
dtype0k
Identity_74IdentityRead_37/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:
a
Identity_75IdentityIdentity_74:output:0"/device:CPU:0*
T0*
_output_shapes
:
|
Read_38/DisableCopyOnReadDisableCopyOnRead'read_38_disablecopyonread_logvar_kernel"/device:CPU:0*
_output_shapes
 �
Read_38/ReadVariableOpReadVariableOp'read_38_disablecopyonread_logvar_kernel^Read_38/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:	�
*
dtype0p
Identity_76IdentityRead_38/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:	�
f
Identity_77IdentityIdentity_76:output:0"/device:CPU:0*
T0*
_output_shapes
:	�
z
Read_39/DisableCopyOnReadDisableCopyOnRead%read_39_disablecopyonread_logvar_bias"/device:CPU:0*
_output_shapes
 �
Read_39/ReadVariableOpReadVariableOp%read_39_disablecopyonread_logvar_bias^Read_39/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:
*
dtype0k
Identity_78IdentityRead_39/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:
a
Identity_79IdentityIdentity_78:output:0"/device:CPU:0*
T0*
_output_shapes
:
�
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:)*
dtype0*�
value�B�)B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-1/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-1/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-1/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-3/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-3/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-3/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-5/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-5/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-5/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-7/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-7/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-7/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-7/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-8/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-8/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-9/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-9/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-9/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-9/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-10/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-10/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-11/gamma/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-11/beta/.ATTRIBUTES/VARIABLE_VALUEB<layer_with_weights-11/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-11/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-12/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-12/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-13/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-13/bias/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH�
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:)*
dtype0*e
value\BZ)B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B �
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0Identity_1:output:0Identity_3:output:0Identity_5:output:0Identity_7:output:0Identity_9:output:0Identity_11:output:0Identity_13:output:0Identity_15:output:0Identity_17:output:0Identity_19:output:0Identity_21:output:0Identity_23:output:0Identity_25:output:0Identity_27:output:0Identity_29:output:0Identity_31:output:0Identity_33:output:0Identity_35:output:0Identity_37:output:0Identity_39:output:0Identity_41:output:0Identity_43:output:0Identity_45:output:0Identity_47:output:0Identity_49:output:0Identity_51:output:0Identity_53:output:0Identity_55:output:0Identity_57:output:0Identity_59:output:0Identity_61:output:0Identity_63:output:0Identity_65:output:0Identity_67:output:0Identity_69:output:0Identity_71:output:0Identity_73:output:0Identity_75:output:0Identity_77:output:0Identity_79:output:0savev2_const"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *7
dtypes-
+2)�
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:�
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 i
Identity_80Identityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: U
Identity_81IdentityIdentity_80:output:0^NoOp*
T0*
_output_shapes
: �
NoOpNoOp^MergeV2Checkpoints^Read/DisableCopyOnRead^Read/ReadVariableOp^Read_1/DisableCopyOnRead^Read_1/ReadVariableOp^Read_10/DisableCopyOnRead^Read_10/ReadVariableOp^Read_11/DisableCopyOnRead^Read_11/ReadVariableOp^Read_12/DisableCopyOnRead^Read_12/ReadVariableOp^Read_13/DisableCopyOnRead^Read_13/ReadVariableOp^Read_14/DisableCopyOnRead^Read_14/ReadVariableOp^Read_15/DisableCopyOnRead^Read_15/ReadVariableOp^Read_16/DisableCopyOnRead^Read_16/ReadVariableOp^Read_17/DisableCopyOnRead^Read_17/ReadVariableOp^Read_18/DisableCopyOnRead^Read_18/ReadVariableOp^Read_19/DisableCopyOnRead^Read_19/ReadVariableOp^Read_2/DisableCopyOnRead^Read_2/ReadVariableOp^Read_20/DisableCopyOnRead^Read_20/ReadVariableOp^Read_21/DisableCopyOnRead^Read_21/ReadVariableOp^Read_22/DisableCopyOnRead^Read_22/ReadVariableOp^Read_23/DisableCopyOnRead^Read_23/ReadVariableOp^Read_24/DisableCopyOnRead^Read_24/ReadVariableOp^Read_25/DisableCopyOnRead^Read_25/ReadVariableOp^Read_26/DisableCopyOnRead^Read_26/ReadVariableOp^Read_27/DisableCopyOnRead^Read_27/ReadVariableOp^Read_28/DisableCopyOnRead^Read_28/ReadVariableOp^Read_29/DisableCopyOnRead^Read_29/ReadVariableOp^Read_3/DisableCopyOnRead^Read_3/ReadVariableOp^Read_30/DisableCopyOnRead^Read_30/ReadVariableOp^Read_31/DisableCopyOnRead^Read_31/ReadVariableOp^Read_32/DisableCopyOnRead^Read_32/ReadVariableOp^Read_33/DisableCopyOnRead^Read_33/ReadVariableOp^Read_34/DisableCopyOnRead^Read_34/ReadVariableOp^Read_35/DisableCopyOnRead^Read_35/ReadVariableOp^Read_36/DisableCopyOnRead^Read_36/ReadVariableOp^Read_37/DisableCopyOnRead^Read_37/ReadVariableOp^Read_38/DisableCopyOnRead^Read_38/ReadVariableOp^Read_39/DisableCopyOnRead^Read_39/ReadVariableOp^Read_4/DisableCopyOnRead^Read_4/ReadVariableOp^Read_5/DisableCopyOnRead^Read_5/ReadVariableOp^Read_6/DisableCopyOnRead^Read_6/ReadVariableOp^Read_7/DisableCopyOnRead^Read_7/ReadVariableOp^Read_8/DisableCopyOnRead^Read_8/ReadVariableOp^Read_9/DisableCopyOnRead^Read_9/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "#
identity_81Identity_81:output:0*g
_input_shapesV
T: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2(
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
Read_33/ReadVariableOpRead_33/ReadVariableOp26
Read_34/DisableCopyOnReadRead_34/DisableCopyOnRead20
Read_34/ReadVariableOpRead_34/ReadVariableOp26
Read_35/DisableCopyOnReadRead_35/DisableCopyOnRead20
Read_35/ReadVariableOpRead_35/ReadVariableOp26
Read_36/DisableCopyOnReadRead_36/DisableCopyOnRead20
Read_36/ReadVariableOpRead_36/ReadVariableOp26
Read_37/DisableCopyOnReadRead_37/DisableCopyOnRead20
Read_37/ReadVariableOpRead_37/ReadVariableOp26
Read_38/DisableCopyOnReadRead_38/DisableCopyOnRead20
Read_38/ReadVariableOpRead_38/ReadVariableOp26
Read_39/DisableCopyOnReadRead_39/DisableCopyOnRead20
Read_39/ReadVariableOpRead_39/ReadVariableOp24
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
_user_specified_namefile_prefix:)

_output_shapes
: 
�
�	
'__inference_encoder_layer_call_fn_12568

inputs!
unknown: 
	unknown_0: 
	unknown_1: 
	unknown_2: 
	unknown_3: 
	unknown_4: #
	unknown_5: @
	unknown_6:@
	unknown_7:@
	unknown_8:@
	unknown_9:@

unknown_10:@$

unknown_11:@@

unknown_12:@

unknown_13:@

unknown_14:@

unknown_15:@

unknown_16:@$

unknown_17:@@

unknown_18:@

unknown_19:@

unknown_20:@

unknown_21:@

unknown_22:@$

unknown_23:@@

unknown_24:@

unknown_25:@

unknown_26:@

unknown_27:@

unknown_28:@$

unknown_29:@ 

unknown_30: 

unknown_31: 

unknown_32: 

unknown_33: 

unknown_34: 

unknown_35:	�


unknown_36:


unknown_37:	�


unknown_38:

identity

identity_1��StatefulPartitionedCall�
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
unknown_32
unknown_33
unknown_34
unknown_35
unknown_36
unknown_37
unknown_38*4
Tin-
+2)*
Tout
2*
_collective_manager_ids
 *:
_output_shapes(
&:���������
:���������
*>
_read_only_resource_inputs 
	
 !"%&'(*-
config_proto

CPU

GPU 2J 8� *K
fFRD
B__inference_encoder_layer_call_and_return_conditional_losses_11818o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������
q

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*'
_output_shapes
:���������
`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*~
_input_shapesm
k:���������: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:���������
 
_user_specified_nameinputs
�	
�
A__inference_logVar_layer_call_and_return_conditional_losses_13540

inputs1
matmul_readvariableop_resource:	�
-
biasadd_readvariableop_resource:

identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	�
*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:
*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������
_
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:���������
w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
]
A__inference_lReLU4_layer_call_and_return_conditional_losses_13309

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
>__inference_bn1_layer_call_and_return_conditional_losses_13026

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

�
B__inference_conv2D6_layer_call_and_return_conditional_losses_11539

inputs8
conv2d_readvariableop_resource:@ -
biasadd_readvariableop_resource: 
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@ *
dtype0�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:��������� *
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:��������� g
IdentityIdentityBiasAdd:output:0^NoOp*
T0*/
_output_shapes
:��������� w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:���������@
 
_user_specified_nameinputs
�
]
A__inference_lReLU5_layer_call_and_return_conditional_losses_13400

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
�
]
A__inference_lReLU2_layer_call_and_return_conditional_losses_13127

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
�
^
B__inference_flatten_layer_call_and_return_conditional_losses_13502

inputs
identityV
ConstConst*
_output_shapes
:*
dtype0*
valueB"����   ]
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:����������Y
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:����������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:��������� :W S
/
_output_shapes
:��������� 
 
_user_specified_nameinputs
�

�
B__inference_conv2D1_layer_call_and_return_conditional_losses_11379

inputs8
conv2d_readvariableop_resource: -
biasadd_readvariableop_resource: 
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: *
dtype0�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:��������� *
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:��������� g
IdentityIdentityBiasAdd:output:0^NoOp*
T0*/
_output_shapes
:��������� w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:���������
 
_user_specified_nameinputs
�
]
A__inference_lReLU6_layer_call_and_return_conditional_losses_11559

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
�
�
&__inference_logVar_layer_call_fn_13530

inputs
unknown:	�

	unknown_0:

identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������
*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *J
fERC
A__inference_logVar_layer_call_and_return_conditional_losses_11579o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������
`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�

�
B__inference_conv2D4_layer_call_and_return_conditional_losses_13237

inputs8
conv2d_readvariableop_resource:@@-
biasadd_readvariableop_resource:@
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype0�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@*
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@g
IdentityIdentityBiasAdd:output:0^NoOp*
T0*/
_output_shapes
:���������@w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:���������@
 
_user_specified_nameinputs
�
�
>__inference_bn1_layer_call_and_return_conditional_losses_13008

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
�[
�
B__inference_encoder_layer_call_and_return_conditional_losses_11603
input_layer'
conv2d1_11380: 
conv2d1_11382: 
	bn1_11385: 
	bn1_11387: 
	bn1_11389: 
	bn1_11391: '
conv2d2_11412: @
conv2d2_11414:@
	bn2_11417:@
	bn2_11419:@
	bn2_11421:@
	bn2_11423:@'
conv2d3_11444:@@
conv2d3_11446:@
	bn3_11449:@
	bn3_11451:@
	bn3_11453:@
	bn3_11455:@'
conv2d4_11476:@@
conv2d4_11478:@
	bn4_11481:@
	bn4_11483:@
	bn4_11485:@
	bn4_11487:@'
conv2d5_11508:@@
conv2d5_11510:@
	bn5_11513:@
	bn5_11515:@
	bn5_11517:@
	bn5_11519:@'
conv2d6_11540:@ 
conv2d6_11542: 
	bn6_11545: 
	bn6_11547: 
	bn6_11549: 
	bn6_11551: 
logvar_11580:	�

logvar_11582:


mean_11596:	�


mean_11598:

identity

identity_1��bn1/StatefulPartitionedCall�bn2/StatefulPartitionedCall�bn3/StatefulPartitionedCall�bn4/StatefulPartitionedCall�bn5/StatefulPartitionedCall�bn6/StatefulPartitionedCall�conv2D1/StatefulPartitionedCall�conv2D2/StatefulPartitionedCall�conv2D3/StatefulPartitionedCall�conv2D4/StatefulPartitionedCall�conv2D5/StatefulPartitionedCall�conv2D6/StatefulPartitionedCall�logVar/StatefulPartitionedCall�mean/StatefulPartitionedCall�
conv2D1/StatefulPartitionedCallStatefulPartitionedCallinput_layerconv2d1_11380conv2d1_11382*
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
GPU 2J 8� *K
fFRD
B__inference_conv2D1_layer_call_and_return_conditional_losses_11379�
bn1/StatefulPartitionedCallStatefulPartitionedCall(conv2D1/StatefulPartitionedCall:output:0	bn1_11385	bn1_11387	bn1_11389	bn1_11391*
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
>__inference_bn1_layer_call_and_return_conditional_losses_11000�
lReLU1/PartitionedCallPartitionedCall$bn1/StatefulPartitionedCall:output:0*
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
A__inference_lReLU1_layer_call_and_return_conditional_losses_11399�
conv2D2/StatefulPartitionedCallStatefulPartitionedCalllReLU1/PartitionedCall:output:0conv2d2_11412conv2d2_11414*
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
GPU 2J 8� *K
fFRD
B__inference_conv2D2_layer_call_and_return_conditional_losses_11411�
bn2/StatefulPartitionedCallStatefulPartitionedCall(conv2D2/StatefulPartitionedCall:output:0	bn2_11417	bn2_11419	bn2_11421	bn2_11423*
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
>__inference_bn2_layer_call_and_return_conditional_losses_11064�
lReLU2/PartitionedCallPartitionedCall$bn2/StatefulPartitionedCall:output:0*
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
A__inference_lReLU2_layer_call_and_return_conditional_losses_11431�
conv2D3/StatefulPartitionedCallStatefulPartitionedCalllReLU2/PartitionedCall:output:0conv2d3_11444conv2d3_11446*
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
GPU 2J 8� *K
fFRD
B__inference_conv2D3_layer_call_and_return_conditional_losses_11443�
bn3/StatefulPartitionedCallStatefulPartitionedCall(conv2D3/StatefulPartitionedCall:output:0	bn3_11449	bn3_11451	bn3_11453	bn3_11455*
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
>__inference_bn3_layer_call_and_return_conditional_losses_11128�
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
A__inference_lReLU3_layer_call_and_return_conditional_losses_11463�
conv2D4/StatefulPartitionedCallStatefulPartitionedCalllReLU3/PartitionedCall:output:0conv2d4_11476conv2d4_11478*
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
GPU 2J 8� *K
fFRD
B__inference_conv2D4_layer_call_and_return_conditional_losses_11475�
bn4/StatefulPartitionedCallStatefulPartitionedCall(conv2D4/StatefulPartitionedCall:output:0	bn4_11481	bn4_11483	bn4_11485	bn4_11487*
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
>__inference_bn4_layer_call_and_return_conditional_losses_11192�
lReLU4/PartitionedCallPartitionedCall$bn4/StatefulPartitionedCall:output:0*
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
A__inference_lReLU4_layer_call_and_return_conditional_losses_11495�
conv2D5/StatefulPartitionedCallStatefulPartitionedCalllReLU4/PartitionedCall:output:0conv2d5_11508conv2d5_11510*
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
GPU 2J 8� *K
fFRD
B__inference_conv2D5_layer_call_and_return_conditional_losses_11507�
bn5/StatefulPartitionedCallStatefulPartitionedCall(conv2D5/StatefulPartitionedCall:output:0	bn5_11513	bn5_11515	bn5_11517	bn5_11519*
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
>__inference_bn5_layer_call_and_return_conditional_losses_11256�
lReLU5/PartitionedCallPartitionedCall$bn5/StatefulPartitionedCall:output:0*
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
A__inference_lReLU5_layer_call_and_return_conditional_losses_11527�
conv2D6/StatefulPartitionedCallStatefulPartitionedCalllReLU5/PartitionedCall:output:0conv2d6_11540conv2d6_11542*
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
GPU 2J 8� *K
fFRD
B__inference_conv2D6_layer_call_and_return_conditional_losses_11539�
bn6/StatefulPartitionedCallStatefulPartitionedCall(conv2D6/StatefulPartitionedCall:output:0	bn6_11545	bn6_11547	bn6_11549	bn6_11551*
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
>__inference_bn6_layer_call_and_return_conditional_losses_11320�
lReLU6/PartitionedCallPartitionedCall$bn6/StatefulPartitionedCall:output:0*
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
A__inference_lReLU6_layer_call_and_return_conditional_losses_11559�
flatten/PartitionedCallPartitionedCalllReLU6/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *K
fFRD
B__inference_flatten_layer_call_and_return_conditional_losses_11567�
logVar/StatefulPartitionedCallStatefulPartitionedCall flatten/PartitionedCall:output:0logvar_11580logvar_11582*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������
*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *J
fERC
A__inference_logVar_layer_call_and_return_conditional_losses_11579�
mean/StatefulPartitionedCallStatefulPartitionedCall flatten/PartitionedCall:output:0
mean_11596
mean_11598*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������
*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *H
fCRA
?__inference_mean_layer_call_and_return_conditional_losses_11595t
IdentityIdentity%mean/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������
x

Identity_1Identity'logVar/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������
�
NoOpNoOp^bn1/StatefulPartitionedCall^bn2/StatefulPartitionedCall^bn3/StatefulPartitionedCall^bn4/StatefulPartitionedCall^bn5/StatefulPartitionedCall^bn6/StatefulPartitionedCall ^conv2D1/StatefulPartitionedCall ^conv2D2/StatefulPartitionedCall ^conv2D3/StatefulPartitionedCall ^conv2D4/StatefulPartitionedCall ^conv2D5/StatefulPartitionedCall ^conv2D6/StatefulPartitionedCall^logVar/StatefulPartitionedCall^mean/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*~
_input_shapesm
k:���������: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2:
bn1/StatefulPartitionedCallbn1/StatefulPartitionedCall2:
bn2/StatefulPartitionedCallbn2/StatefulPartitionedCall2:
bn3/StatefulPartitionedCallbn3/StatefulPartitionedCall2:
bn4/StatefulPartitionedCallbn4/StatefulPartitionedCall2:
bn5/StatefulPartitionedCallbn5/StatefulPartitionedCall2:
bn6/StatefulPartitionedCallbn6/StatefulPartitionedCall2B
conv2D1/StatefulPartitionedCallconv2D1/StatefulPartitionedCall2B
conv2D2/StatefulPartitionedCallconv2D2/StatefulPartitionedCall2B
conv2D3/StatefulPartitionedCallconv2D3/StatefulPartitionedCall2B
conv2D4/StatefulPartitionedCallconv2D4/StatefulPartitionedCall2B
conv2D5/StatefulPartitionedCallconv2D5/StatefulPartitionedCall2B
conv2D6/StatefulPartitionedCallconv2D6/StatefulPartitionedCall2@
logVar/StatefulPartitionedCalllogVar/StatefulPartitionedCall2<
mean/StatefulPartitionedCallmean/StatefulPartitionedCall:\ X
/
_output_shapes
:���������
%
_user_specified_nameinput_layer
�
� 
 __inference__wrapped_model_10981
input_layerH
.encoder_conv2d1_conv2d_readvariableop_resource: =
/encoder_conv2d1_biasadd_readvariableop_resource: 1
#encoder_bn1_readvariableop_resource: 3
%encoder_bn1_readvariableop_1_resource: B
4encoder_bn1_fusedbatchnormv3_readvariableop_resource: D
6encoder_bn1_fusedbatchnormv3_readvariableop_1_resource: H
.encoder_conv2d2_conv2d_readvariableop_resource: @=
/encoder_conv2d2_biasadd_readvariableop_resource:@1
#encoder_bn2_readvariableop_resource:@3
%encoder_bn2_readvariableop_1_resource:@B
4encoder_bn2_fusedbatchnormv3_readvariableop_resource:@D
6encoder_bn2_fusedbatchnormv3_readvariableop_1_resource:@H
.encoder_conv2d3_conv2d_readvariableop_resource:@@=
/encoder_conv2d3_biasadd_readvariableop_resource:@1
#encoder_bn3_readvariableop_resource:@3
%encoder_bn3_readvariableop_1_resource:@B
4encoder_bn3_fusedbatchnormv3_readvariableop_resource:@D
6encoder_bn3_fusedbatchnormv3_readvariableop_1_resource:@H
.encoder_conv2d4_conv2d_readvariableop_resource:@@=
/encoder_conv2d4_biasadd_readvariableop_resource:@1
#encoder_bn4_readvariableop_resource:@3
%encoder_bn4_readvariableop_1_resource:@B
4encoder_bn4_fusedbatchnormv3_readvariableop_resource:@D
6encoder_bn4_fusedbatchnormv3_readvariableop_1_resource:@H
.encoder_conv2d5_conv2d_readvariableop_resource:@@=
/encoder_conv2d5_biasadd_readvariableop_resource:@1
#encoder_bn5_readvariableop_resource:@3
%encoder_bn5_readvariableop_1_resource:@B
4encoder_bn5_fusedbatchnormv3_readvariableop_resource:@D
6encoder_bn5_fusedbatchnormv3_readvariableop_1_resource:@H
.encoder_conv2d6_conv2d_readvariableop_resource:@ =
/encoder_conv2d6_biasadd_readvariableop_resource: 1
#encoder_bn6_readvariableop_resource: 3
%encoder_bn6_readvariableop_1_resource: B
4encoder_bn6_fusedbatchnormv3_readvariableop_resource: D
6encoder_bn6_fusedbatchnormv3_readvariableop_1_resource: @
-encoder_logvar_matmul_readvariableop_resource:	�
<
.encoder_logvar_biasadd_readvariableop_resource:
>
+encoder_mean_matmul_readvariableop_resource:	�
:
,encoder_mean_biasadd_readvariableop_resource:

identity

identity_1��+encoder/bn1/FusedBatchNormV3/ReadVariableOp�-encoder/bn1/FusedBatchNormV3/ReadVariableOp_1�encoder/bn1/ReadVariableOp�encoder/bn1/ReadVariableOp_1�+encoder/bn2/FusedBatchNormV3/ReadVariableOp�-encoder/bn2/FusedBatchNormV3/ReadVariableOp_1�encoder/bn2/ReadVariableOp�encoder/bn2/ReadVariableOp_1�+encoder/bn3/FusedBatchNormV3/ReadVariableOp�-encoder/bn3/FusedBatchNormV3/ReadVariableOp_1�encoder/bn3/ReadVariableOp�encoder/bn3/ReadVariableOp_1�+encoder/bn4/FusedBatchNormV3/ReadVariableOp�-encoder/bn4/FusedBatchNormV3/ReadVariableOp_1�encoder/bn4/ReadVariableOp�encoder/bn4/ReadVariableOp_1�+encoder/bn5/FusedBatchNormV3/ReadVariableOp�-encoder/bn5/FusedBatchNormV3/ReadVariableOp_1�encoder/bn5/ReadVariableOp�encoder/bn5/ReadVariableOp_1�+encoder/bn6/FusedBatchNormV3/ReadVariableOp�-encoder/bn6/FusedBatchNormV3/ReadVariableOp_1�encoder/bn6/ReadVariableOp�encoder/bn6/ReadVariableOp_1�&encoder/conv2D1/BiasAdd/ReadVariableOp�%encoder/conv2D1/Conv2D/ReadVariableOp�&encoder/conv2D2/BiasAdd/ReadVariableOp�%encoder/conv2D2/Conv2D/ReadVariableOp�&encoder/conv2D3/BiasAdd/ReadVariableOp�%encoder/conv2D3/Conv2D/ReadVariableOp�&encoder/conv2D4/BiasAdd/ReadVariableOp�%encoder/conv2D4/Conv2D/ReadVariableOp�&encoder/conv2D5/BiasAdd/ReadVariableOp�%encoder/conv2D5/Conv2D/ReadVariableOp�&encoder/conv2D6/BiasAdd/ReadVariableOp�%encoder/conv2D6/Conv2D/ReadVariableOp�%encoder/logVar/BiasAdd/ReadVariableOp�$encoder/logVar/MatMul/ReadVariableOp�#encoder/mean/BiasAdd/ReadVariableOp�"encoder/mean/MatMul/ReadVariableOp�
%encoder/conv2D1/Conv2D/ReadVariableOpReadVariableOp.encoder_conv2d1_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype0�
encoder/conv2D1/Conv2DConv2Dinput_layer-encoder/conv2D1/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:��������� *
paddingSAME*
strides
�
&encoder/conv2D1/BiasAdd/ReadVariableOpReadVariableOp/encoder_conv2d1_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0�
encoder/conv2D1/BiasAddBiasAddencoder/conv2D1/Conv2D:output:0.encoder/conv2D1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:��������� z
encoder/bn1/ReadVariableOpReadVariableOp#encoder_bn1_readvariableop_resource*
_output_shapes
: *
dtype0~
encoder/bn1/ReadVariableOp_1ReadVariableOp%encoder_bn1_readvariableop_1_resource*
_output_shapes
: *
dtype0�
+encoder/bn1/FusedBatchNormV3/ReadVariableOpReadVariableOp4encoder_bn1_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype0�
-encoder/bn1/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp6encoder_bn1_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype0�
encoder/bn1/FusedBatchNormV3FusedBatchNormV3 encoder/conv2D1/BiasAdd:output:0"encoder/bn1/ReadVariableOp:value:0$encoder/bn1/ReadVariableOp_1:value:03encoder/bn1/FusedBatchNormV3/ReadVariableOp:value:05encoder/bn1/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:��������� : : : : :*
epsilon%o�:*
is_training( �
encoder/lReLU1/LeakyRelu	LeakyRelu encoder/bn1/FusedBatchNormV3:y:0*/
_output_shapes
:��������� *
alpha%���>�
%encoder/conv2D2/Conv2D/ReadVariableOpReadVariableOp.encoder_conv2d2_conv2d_readvariableop_resource*&
_output_shapes
: @*
dtype0�
encoder/conv2D2/Conv2DConv2D&encoder/lReLU1/LeakyRelu:activations:0-encoder/conv2D2/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@*
paddingSAME*
strides
�
&encoder/conv2D2/BiasAdd/ReadVariableOpReadVariableOp/encoder_conv2d2_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0�
encoder/conv2D2/BiasAddBiasAddencoder/conv2D2/Conv2D:output:0.encoder/conv2D2/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@z
encoder/bn2/ReadVariableOpReadVariableOp#encoder_bn2_readvariableop_resource*
_output_shapes
:@*
dtype0~
encoder/bn2/ReadVariableOp_1ReadVariableOp%encoder_bn2_readvariableop_1_resource*
_output_shapes
:@*
dtype0�
+encoder/bn2/FusedBatchNormV3/ReadVariableOpReadVariableOp4encoder_bn2_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype0�
-encoder/bn2/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp6encoder_bn2_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype0�
encoder/bn2/FusedBatchNormV3FusedBatchNormV3 encoder/conv2D2/BiasAdd:output:0"encoder/bn2/ReadVariableOp:value:0$encoder/bn2/ReadVariableOp_1:value:03encoder/bn2/FusedBatchNormV3/ReadVariableOp:value:05encoder/bn2/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:���������@:@:@:@:@:*
epsilon%o�:*
is_training( �
encoder/lReLU2/LeakyRelu	LeakyRelu encoder/bn2/FusedBatchNormV3:y:0*/
_output_shapes
:���������@*
alpha%���>�
%encoder/conv2D3/Conv2D/ReadVariableOpReadVariableOp.encoder_conv2d3_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype0�
encoder/conv2D3/Conv2DConv2D&encoder/lReLU2/LeakyRelu:activations:0-encoder/conv2D3/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@*
paddingSAME*
strides
�
&encoder/conv2D3/BiasAdd/ReadVariableOpReadVariableOp/encoder_conv2d3_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0�
encoder/conv2D3/BiasAddBiasAddencoder/conv2D3/Conv2D:output:0.encoder/conv2D3/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@z
encoder/bn3/ReadVariableOpReadVariableOp#encoder_bn3_readvariableop_resource*
_output_shapes
:@*
dtype0~
encoder/bn3/ReadVariableOp_1ReadVariableOp%encoder_bn3_readvariableop_1_resource*
_output_shapes
:@*
dtype0�
+encoder/bn3/FusedBatchNormV3/ReadVariableOpReadVariableOp4encoder_bn3_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype0�
-encoder/bn3/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp6encoder_bn3_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype0�
encoder/bn3/FusedBatchNormV3FusedBatchNormV3 encoder/conv2D3/BiasAdd:output:0"encoder/bn3/ReadVariableOp:value:0$encoder/bn3/ReadVariableOp_1:value:03encoder/bn3/FusedBatchNormV3/ReadVariableOp:value:05encoder/bn3/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:���������@:@:@:@:@:*
epsilon%o�:*
is_training( �
encoder/lReLU3/LeakyRelu	LeakyRelu encoder/bn3/FusedBatchNormV3:y:0*/
_output_shapes
:���������@*
alpha%���>�
%encoder/conv2D4/Conv2D/ReadVariableOpReadVariableOp.encoder_conv2d4_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype0�
encoder/conv2D4/Conv2DConv2D&encoder/lReLU3/LeakyRelu:activations:0-encoder/conv2D4/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@*
paddingSAME*
strides
�
&encoder/conv2D4/BiasAdd/ReadVariableOpReadVariableOp/encoder_conv2d4_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0�
encoder/conv2D4/BiasAddBiasAddencoder/conv2D4/Conv2D:output:0.encoder/conv2D4/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@z
encoder/bn4/ReadVariableOpReadVariableOp#encoder_bn4_readvariableop_resource*
_output_shapes
:@*
dtype0~
encoder/bn4/ReadVariableOp_1ReadVariableOp%encoder_bn4_readvariableop_1_resource*
_output_shapes
:@*
dtype0�
+encoder/bn4/FusedBatchNormV3/ReadVariableOpReadVariableOp4encoder_bn4_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype0�
-encoder/bn4/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp6encoder_bn4_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype0�
encoder/bn4/FusedBatchNormV3FusedBatchNormV3 encoder/conv2D4/BiasAdd:output:0"encoder/bn4/ReadVariableOp:value:0$encoder/bn4/ReadVariableOp_1:value:03encoder/bn4/FusedBatchNormV3/ReadVariableOp:value:05encoder/bn4/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:���������@:@:@:@:@:*
epsilon%o�:*
is_training( �
encoder/lReLU4/LeakyRelu	LeakyRelu encoder/bn4/FusedBatchNormV3:y:0*/
_output_shapes
:���������@*
alpha%���>�
%encoder/conv2D5/Conv2D/ReadVariableOpReadVariableOp.encoder_conv2d5_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype0�
encoder/conv2D5/Conv2DConv2D&encoder/lReLU4/LeakyRelu:activations:0-encoder/conv2D5/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@*
paddingSAME*
strides
�
&encoder/conv2D5/BiasAdd/ReadVariableOpReadVariableOp/encoder_conv2d5_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0�
encoder/conv2D5/BiasAddBiasAddencoder/conv2D5/Conv2D:output:0.encoder/conv2D5/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@z
encoder/bn5/ReadVariableOpReadVariableOp#encoder_bn5_readvariableop_resource*
_output_shapes
:@*
dtype0~
encoder/bn5/ReadVariableOp_1ReadVariableOp%encoder_bn5_readvariableop_1_resource*
_output_shapes
:@*
dtype0�
+encoder/bn5/FusedBatchNormV3/ReadVariableOpReadVariableOp4encoder_bn5_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype0�
-encoder/bn5/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp6encoder_bn5_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype0�
encoder/bn5/FusedBatchNormV3FusedBatchNormV3 encoder/conv2D5/BiasAdd:output:0"encoder/bn5/ReadVariableOp:value:0$encoder/bn5/ReadVariableOp_1:value:03encoder/bn5/FusedBatchNormV3/ReadVariableOp:value:05encoder/bn5/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:���������@:@:@:@:@:*
epsilon%o�:*
is_training( �
encoder/lReLU5/LeakyRelu	LeakyRelu encoder/bn5/FusedBatchNormV3:y:0*/
_output_shapes
:���������@*
alpha%���>�
%encoder/conv2D6/Conv2D/ReadVariableOpReadVariableOp.encoder_conv2d6_conv2d_readvariableop_resource*&
_output_shapes
:@ *
dtype0�
encoder/conv2D6/Conv2DConv2D&encoder/lReLU5/LeakyRelu:activations:0-encoder/conv2D6/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:��������� *
paddingSAME*
strides
�
&encoder/conv2D6/BiasAdd/ReadVariableOpReadVariableOp/encoder_conv2d6_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0�
encoder/conv2D6/BiasAddBiasAddencoder/conv2D6/Conv2D:output:0.encoder/conv2D6/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:��������� z
encoder/bn6/ReadVariableOpReadVariableOp#encoder_bn6_readvariableop_resource*
_output_shapes
: *
dtype0~
encoder/bn6/ReadVariableOp_1ReadVariableOp%encoder_bn6_readvariableop_1_resource*
_output_shapes
: *
dtype0�
+encoder/bn6/FusedBatchNormV3/ReadVariableOpReadVariableOp4encoder_bn6_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype0�
-encoder/bn6/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp6encoder_bn6_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype0�
encoder/bn6/FusedBatchNormV3FusedBatchNormV3 encoder/conv2D6/BiasAdd:output:0"encoder/bn6/ReadVariableOp:value:0$encoder/bn6/ReadVariableOp_1:value:03encoder/bn6/FusedBatchNormV3/ReadVariableOp:value:05encoder/bn6/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:��������� : : : : :*
epsilon%o�:*
is_training( �
encoder/lReLU6/LeakyRelu	LeakyRelu encoder/bn6/FusedBatchNormV3:y:0*/
_output_shapes
:��������� *
alpha%���>f
encoder/flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"����   �
encoder/flatten/ReshapeReshape&encoder/lReLU6/LeakyRelu:activations:0encoder/flatten/Const:output:0*
T0*(
_output_shapes
:�����������
$encoder/logVar/MatMul/ReadVariableOpReadVariableOp-encoder_logvar_matmul_readvariableop_resource*
_output_shapes
:	�
*
dtype0�
encoder/logVar/MatMulMatMul encoder/flatten/Reshape:output:0,encoder/logVar/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������
�
%encoder/logVar/BiasAdd/ReadVariableOpReadVariableOp.encoder_logvar_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype0�
encoder/logVar/BiasAddBiasAddencoder/logVar/MatMul:product:0-encoder/logVar/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������
�
"encoder/mean/MatMul/ReadVariableOpReadVariableOp+encoder_mean_matmul_readvariableop_resource*
_output_shapes
:	�
*
dtype0�
encoder/mean/MatMulMatMul encoder/flatten/Reshape:output:0*encoder/mean/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������
�
#encoder/mean/BiasAdd/ReadVariableOpReadVariableOp,encoder_mean_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype0�
encoder/mean/BiasAddBiasAddencoder/mean/MatMul:product:0+encoder/mean/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������
n
IdentityIdentityencoder/logVar/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:���������
n

Identity_1Identityencoder/mean/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:���������
�
NoOpNoOp,^encoder/bn1/FusedBatchNormV3/ReadVariableOp.^encoder/bn1/FusedBatchNormV3/ReadVariableOp_1^encoder/bn1/ReadVariableOp^encoder/bn1/ReadVariableOp_1,^encoder/bn2/FusedBatchNormV3/ReadVariableOp.^encoder/bn2/FusedBatchNormV3/ReadVariableOp_1^encoder/bn2/ReadVariableOp^encoder/bn2/ReadVariableOp_1,^encoder/bn3/FusedBatchNormV3/ReadVariableOp.^encoder/bn3/FusedBatchNormV3/ReadVariableOp_1^encoder/bn3/ReadVariableOp^encoder/bn3/ReadVariableOp_1,^encoder/bn4/FusedBatchNormV3/ReadVariableOp.^encoder/bn4/FusedBatchNormV3/ReadVariableOp_1^encoder/bn4/ReadVariableOp^encoder/bn4/ReadVariableOp_1,^encoder/bn5/FusedBatchNormV3/ReadVariableOp.^encoder/bn5/FusedBatchNormV3/ReadVariableOp_1^encoder/bn5/ReadVariableOp^encoder/bn5/ReadVariableOp_1,^encoder/bn6/FusedBatchNormV3/ReadVariableOp.^encoder/bn6/FusedBatchNormV3/ReadVariableOp_1^encoder/bn6/ReadVariableOp^encoder/bn6/ReadVariableOp_1'^encoder/conv2D1/BiasAdd/ReadVariableOp&^encoder/conv2D1/Conv2D/ReadVariableOp'^encoder/conv2D2/BiasAdd/ReadVariableOp&^encoder/conv2D2/Conv2D/ReadVariableOp'^encoder/conv2D3/BiasAdd/ReadVariableOp&^encoder/conv2D3/Conv2D/ReadVariableOp'^encoder/conv2D4/BiasAdd/ReadVariableOp&^encoder/conv2D4/Conv2D/ReadVariableOp'^encoder/conv2D5/BiasAdd/ReadVariableOp&^encoder/conv2D5/Conv2D/ReadVariableOp'^encoder/conv2D6/BiasAdd/ReadVariableOp&^encoder/conv2D6/Conv2D/ReadVariableOp&^encoder/logVar/BiasAdd/ReadVariableOp%^encoder/logVar/MatMul/ReadVariableOp$^encoder/mean/BiasAdd/ReadVariableOp#^encoder/mean/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*~
_input_shapesm
k:���������: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2Z
+encoder/bn1/FusedBatchNormV3/ReadVariableOp+encoder/bn1/FusedBatchNormV3/ReadVariableOp2^
-encoder/bn1/FusedBatchNormV3/ReadVariableOp_1-encoder/bn1/FusedBatchNormV3/ReadVariableOp_128
encoder/bn1/ReadVariableOpencoder/bn1/ReadVariableOp2<
encoder/bn1/ReadVariableOp_1encoder/bn1/ReadVariableOp_12Z
+encoder/bn2/FusedBatchNormV3/ReadVariableOp+encoder/bn2/FusedBatchNormV3/ReadVariableOp2^
-encoder/bn2/FusedBatchNormV3/ReadVariableOp_1-encoder/bn2/FusedBatchNormV3/ReadVariableOp_128
encoder/bn2/ReadVariableOpencoder/bn2/ReadVariableOp2<
encoder/bn2/ReadVariableOp_1encoder/bn2/ReadVariableOp_12Z
+encoder/bn3/FusedBatchNormV3/ReadVariableOp+encoder/bn3/FusedBatchNormV3/ReadVariableOp2^
-encoder/bn3/FusedBatchNormV3/ReadVariableOp_1-encoder/bn3/FusedBatchNormV3/ReadVariableOp_128
encoder/bn3/ReadVariableOpencoder/bn3/ReadVariableOp2<
encoder/bn3/ReadVariableOp_1encoder/bn3/ReadVariableOp_12Z
+encoder/bn4/FusedBatchNormV3/ReadVariableOp+encoder/bn4/FusedBatchNormV3/ReadVariableOp2^
-encoder/bn4/FusedBatchNormV3/ReadVariableOp_1-encoder/bn4/FusedBatchNormV3/ReadVariableOp_128
encoder/bn4/ReadVariableOpencoder/bn4/ReadVariableOp2<
encoder/bn4/ReadVariableOp_1encoder/bn4/ReadVariableOp_12Z
+encoder/bn5/FusedBatchNormV3/ReadVariableOp+encoder/bn5/FusedBatchNormV3/ReadVariableOp2^
-encoder/bn5/FusedBatchNormV3/ReadVariableOp_1-encoder/bn5/FusedBatchNormV3/ReadVariableOp_128
encoder/bn5/ReadVariableOpencoder/bn5/ReadVariableOp2<
encoder/bn5/ReadVariableOp_1encoder/bn5/ReadVariableOp_12Z
+encoder/bn6/FusedBatchNormV3/ReadVariableOp+encoder/bn6/FusedBatchNormV3/ReadVariableOp2^
-encoder/bn6/FusedBatchNormV3/ReadVariableOp_1-encoder/bn6/FusedBatchNormV3/ReadVariableOp_128
encoder/bn6/ReadVariableOpencoder/bn6/ReadVariableOp2<
encoder/bn6/ReadVariableOp_1encoder/bn6/ReadVariableOp_12P
&encoder/conv2D1/BiasAdd/ReadVariableOp&encoder/conv2D1/BiasAdd/ReadVariableOp2N
%encoder/conv2D1/Conv2D/ReadVariableOp%encoder/conv2D1/Conv2D/ReadVariableOp2P
&encoder/conv2D2/BiasAdd/ReadVariableOp&encoder/conv2D2/BiasAdd/ReadVariableOp2N
%encoder/conv2D2/Conv2D/ReadVariableOp%encoder/conv2D2/Conv2D/ReadVariableOp2P
&encoder/conv2D3/BiasAdd/ReadVariableOp&encoder/conv2D3/BiasAdd/ReadVariableOp2N
%encoder/conv2D3/Conv2D/ReadVariableOp%encoder/conv2D3/Conv2D/ReadVariableOp2P
&encoder/conv2D4/BiasAdd/ReadVariableOp&encoder/conv2D4/BiasAdd/ReadVariableOp2N
%encoder/conv2D4/Conv2D/ReadVariableOp%encoder/conv2D4/Conv2D/ReadVariableOp2P
&encoder/conv2D5/BiasAdd/ReadVariableOp&encoder/conv2D5/BiasAdd/ReadVariableOp2N
%encoder/conv2D5/Conv2D/ReadVariableOp%encoder/conv2D5/Conv2D/ReadVariableOp2P
&encoder/conv2D6/BiasAdd/ReadVariableOp&encoder/conv2D6/BiasAdd/ReadVariableOp2N
%encoder/conv2D6/Conv2D/ReadVariableOp%encoder/conv2D6/Conv2D/ReadVariableOp2N
%encoder/logVar/BiasAdd/ReadVariableOp%encoder/logVar/BiasAdd/ReadVariableOp2L
$encoder/logVar/MatMul/ReadVariableOp$encoder/logVar/MatMul/ReadVariableOp2J
#encoder/mean/BiasAdd/ReadVariableOp#encoder/mean/BiasAdd/ReadVariableOp2H
"encoder/mean/MatMul/ReadVariableOp"encoder/mean/MatMul/ReadVariableOp:\ X
/
_output_shapes
:���������
%
_user_specified_nameinput_layer
�
�
>__inference_bn4_layer_call_and_return_conditional_losses_11210

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
�
�
#__inference_bn4_layer_call_fn_13263

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
>__inference_bn4_layer_call_and_return_conditional_losses_11210�
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
�[
�
B__inference_encoder_layer_call_and_return_conditional_losses_11818

inputs'
conv2d1_11715: 
conv2d1_11717: 
	bn1_11720: 
	bn1_11722: 
	bn1_11724: 
	bn1_11726: '
conv2d2_11730: @
conv2d2_11732:@
	bn2_11735:@
	bn2_11737:@
	bn2_11739:@
	bn2_11741:@'
conv2d3_11745:@@
conv2d3_11747:@
	bn3_11750:@
	bn3_11752:@
	bn3_11754:@
	bn3_11756:@'
conv2d4_11760:@@
conv2d4_11762:@
	bn4_11765:@
	bn4_11767:@
	bn4_11769:@
	bn4_11771:@'
conv2d5_11775:@@
conv2d5_11777:@
	bn5_11780:@
	bn5_11782:@
	bn5_11784:@
	bn5_11786:@'
conv2d6_11790:@ 
conv2d6_11792: 
	bn6_11795: 
	bn6_11797: 
	bn6_11799: 
	bn6_11801: 
logvar_11806:	�

logvar_11808:


mean_11811:	�


mean_11813:

identity

identity_1��bn1/StatefulPartitionedCall�bn2/StatefulPartitionedCall�bn3/StatefulPartitionedCall�bn4/StatefulPartitionedCall�bn5/StatefulPartitionedCall�bn6/StatefulPartitionedCall�conv2D1/StatefulPartitionedCall�conv2D2/StatefulPartitionedCall�conv2D3/StatefulPartitionedCall�conv2D4/StatefulPartitionedCall�conv2D5/StatefulPartitionedCall�conv2D6/StatefulPartitionedCall�logVar/StatefulPartitionedCall�mean/StatefulPartitionedCall�
conv2D1/StatefulPartitionedCallStatefulPartitionedCallinputsconv2d1_11715conv2d1_11717*
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
GPU 2J 8� *K
fFRD
B__inference_conv2D1_layer_call_and_return_conditional_losses_11379�
bn1/StatefulPartitionedCallStatefulPartitionedCall(conv2D1/StatefulPartitionedCall:output:0	bn1_11720	bn1_11722	bn1_11724	bn1_11726*
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
>__inference_bn1_layer_call_and_return_conditional_losses_11000�
lReLU1/PartitionedCallPartitionedCall$bn1/StatefulPartitionedCall:output:0*
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
A__inference_lReLU1_layer_call_and_return_conditional_losses_11399�
conv2D2/StatefulPartitionedCallStatefulPartitionedCalllReLU1/PartitionedCall:output:0conv2d2_11730conv2d2_11732*
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
GPU 2J 8� *K
fFRD
B__inference_conv2D2_layer_call_and_return_conditional_losses_11411�
bn2/StatefulPartitionedCallStatefulPartitionedCall(conv2D2/StatefulPartitionedCall:output:0	bn2_11735	bn2_11737	bn2_11739	bn2_11741*
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
>__inference_bn2_layer_call_and_return_conditional_losses_11064�
lReLU2/PartitionedCallPartitionedCall$bn2/StatefulPartitionedCall:output:0*
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
A__inference_lReLU2_layer_call_and_return_conditional_losses_11431�
conv2D3/StatefulPartitionedCallStatefulPartitionedCalllReLU2/PartitionedCall:output:0conv2d3_11745conv2d3_11747*
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
GPU 2J 8� *K
fFRD
B__inference_conv2D3_layer_call_and_return_conditional_losses_11443�
bn3/StatefulPartitionedCallStatefulPartitionedCall(conv2D3/StatefulPartitionedCall:output:0	bn3_11750	bn3_11752	bn3_11754	bn3_11756*
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
>__inference_bn3_layer_call_and_return_conditional_losses_11128�
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
A__inference_lReLU3_layer_call_and_return_conditional_losses_11463�
conv2D4/StatefulPartitionedCallStatefulPartitionedCalllReLU3/PartitionedCall:output:0conv2d4_11760conv2d4_11762*
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
GPU 2J 8� *K
fFRD
B__inference_conv2D4_layer_call_and_return_conditional_losses_11475�
bn4/StatefulPartitionedCallStatefulPartitionedCall(conv2D4/StatefulPartitionedCall:output:0	bn4_11765	bn4_11767	bn4_11769	bn4_11771*
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
>__inference_bn4_layer_call_and_return_conditional_losses_11192�
lReLU4/PartitionedCallPartitionedCall$bn4/StatefulPartitionedCall:output:0*
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
A__inference_lReLU4_layer_call_and_return_conditional_losses_11495�
conv2D5/StatefulPartitionedCallStatefulPartitionedCalllReLU4/PartitionedCall:output:0conv2d5_11775conv2d5_11777*
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
GPU 2J 8� *K
fFRD
B__inference_conv2D5_layer_call_and_return_conditional_losses_11507�
bn5/StatefulPartitionedCallStatefulPartitionedCall(conv2D5/StatefulPartitionedCall:output:0	bn5_11780	bn5_11782	bn5_11784	bn5_11786*
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
>__inference_bn5_layer_call_and_return_conditional_losses_11256�
lReLU5/PartitionedCallPartitionedCall$bn5/StatefulPartitionedCall:output:0*
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
A__inference_lReLU5_layer_call_and_return_conditional_losses_11527�
conv2D6/StatefulPartitionedCallStatefulPartitionedCalllReLU5/PartitionedCall:output:0conv2d6_11790conv2d6_11792*
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
GPU 2J 8� *K
fFRD
B__inference_conv2D6_layer_call_and_return_conditional_losses_11539�
bn6/StatefulPartitionedCallStatefulPartitionedCall(conv2D6/StatefulPartitionedCall:output:0	bn6_11795	bn6_11797	bn6_11799	bn6_11801*
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
>__inference_bn6_layer_call_and_return_conditional_losses_11320�
lReLU6/PartitionedCallPartitionedCall$bn6/StatefulPartitionedCall:output:0*
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
A__inference_lReLU6_layer_call_and_return_conditional_losses_11559�
flatten/PartitionedCallPartitionedCalllReLU6/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *K
fFRD
B__inference_flatten_layer_call_and_return_conditional_losses_11567�
logVar/StatefulPartitionedCallStatefulPartitionedCall flatten/PartitionedCall:output:0logvar_11806logvar_11808*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������
*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *J
fERC
A__inference_logVar_layer_call_and_return_conditional_losses_11579�
mean/StatefulPartitionedCallStatefulPartitionedCall flatten/PartitionedCall:output:0
mean_11811
mean_11813*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������
*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *H
fCRA
?__inference_mean_layer_call_and_return_conditional_losses_11595t
IdentityIdentity%mean/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������
x

Identity_1Identity'logVar/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������
�
NoOpNoOp^bn1/StatefulPartitionedCall^bn2/StatefulPartitionedCall^bn3/StatefulPartitionedCall^bn4/StatefulPartitionedCall^bn5/StatefulPartitionedCall^bn6/StatefulPartitionedCall ^conv2D1/StatefulPartitionedCall ^conv2D2/StatefulPartitionedCall ^conv2D3/StatefulPartitionedCall ^conv2D4/StatefulPartitionedCall ^conv2D5/StatefulPartitionedCall ^conv2D6/StatefulPartitionedCall^logVar/StatefulPartitionedCall^mean/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*~
_input_shapesm
k:���������: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2:
bn1/StatefulPartitionedCallbn1/StatefulPartitionedCall2:
bn2/StatefulPartitionedCallbn2/StatefulPartitionedCall2:
bn3/StatefulPartitionedCallbn3/StatefulPartitionedCall2:
bn4/StatefulPartitionedCallbn4/StatefulPartitionedCall2:
bn5/StatefulPartitionedCallbn5/StatefulPartitionedCall2:
bn6/StatefulPartitionedCallbn6/StatefulPartitionedCall2B
conv2D1/StatefulPartitionedCallconv2D1/StatefulPartitionedCall2B
conv2D2/StatefulPartitionedCallconv2D2/StatefulPartitionedCall2B
conv2D3/StatefulPartitionedCallconv2D3/StatefulPartitionedCall2B
conv2D4/StatefulPartitionedCallconv2D4/StatefulPartitionedCall2B
conv2D5/StatefulPartitionedCallconv2D5/StatefulPartitionedCall2B
conv2D6/StatefulPartitionedCallconv2D6/StatefulPartitionedCall2@
logVar/StatefulPartitionedCalllogVar/StatefulPartitionedCall2<
mean/StatefulPartitionedCallmean/StatefulPartitionedCall:W S
/
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
>__inference_bn2_layer_call_and_return_conditional_losses_13117

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
>__inference_bn2_layer_call_and_return_conditional_losses_11082

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
�
�
'__inference_conv2D3_layer_call_fn_13136

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
 */
_output_shapes
:���������@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *K
fFRD
B__inference_conv2D3_layer_call_and_return_conditional_losses_11443w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:���������@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������@: : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:���������@
 
_user_specified_nameinputs
�
�
>__inference_bn5_layer_call_and_return_conditional_losses_11256

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
?__inference_mean_layer_call_and_return_conditional_losses_11595

inputs1
matmul_readvariableop_resource:	�
-
biasadd_readvariableop_resource:

identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	�
*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:
*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������
_
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:���������
w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�

�
B__inference_conv2D5_layer_call_and_return_conditional_losses_13328

inputs8
conv2d_readvariableop_resource:@@-
biasadd_readvariableop_resource:@
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype0�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@*
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@g
IdentityIdentityBiasAdd:output:0^NoOp*
T0*/
_output_shapes
:���������@w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:���������@
 
_user_specified_nameinputs
�
B
&__inference_lReLU6_layer_call_fn_13486

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
A__inference_lReLU6_layer_call_and_return_conditional_losses_11559h
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
�
]
A__inference_lReLU1_layer_call_and_return_conditional_losses_13036

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
�
�
>__inference_bn5_layer_call_and_return_conditional_losses_13390

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
>__inference_bn2_layer_call_and_return_conditional_losses_13099

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
�
�	
#__inference_signature_wrapper_12481
input_layer!
unknown: 
	unknown_0: 
	unknown_1: 
	unknown_2: 
	unknown_3: 
	unknown_4: #
	unknown_5: @
	unknown_6:@
	unknown_7:@
	unknown_8:@
	unknown_9:@

unknown_10:@$

unknown_11:@@

unknown_12:@

unknown_13:@

unknown_14:@

unknown_15:@

unknown_16:@$

unknown_17:@@

unknown_18:@

unknown_19:@

unknown_20:@

unknown_21:@

unknown_22:@$

unknown_23:@@

unknown_24:@

unknown_25:@

unknown_26:@

unknown_27:@

unknown_28:@$

unknown_29:@ 

unknown_30: 

unknown_31: 

unknown_32: 

unknown_33: 

unknown_34: 

unknown_35:	�


unknown_36:


unknown_37:	�


unknown_38:

identity

identity_1��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinput_layerunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
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
unknown_32
unknown_33
unknown_34
unknown_35
unknown_36
unknown_37
unknown_38*4
Tin-
+2)*
Tout
2*
_collective_manager_ids
 *:
_output_shapes(
&:���������
:���������
*J
_read_only_resource_inputs,
*(	
 !"#$%&'(*-
config_proto

CPU

GPU 2J 8� *)
f$R"
 __inference__wrapped_model_10981o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������
q

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*'
_output_shapes
:���������
`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*~
_input_shapesm
k:���������: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:\ X
/
_output_shapes
:���������
%
_user_specified_nameinput_layer
�
�
>__inference_bn6_layer_call_and_return_conditional_losses_11320

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
#__inference_bn3_layer_call_fn_13159

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
>__inference_bn3_layer_call_and_return_conditional_losses_11128�
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

�
B__inference_conv2D6_layer_call_and_return_conditional_losses_13419

inputs8
conv2d_readvariableop_resource:@ -
biasadd_readvariableop_resource: 
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@ *
dtype0�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:��������� *
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:��������� g
IdentityIdentityBiasAdd:output:0^NoOp*
T0*/
_output_shapes
:��������� w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:���������@
 
_user_specified_nameinputs
�

�
B__inference_conv2D5_layer_call_and_return_conditional_losses_11507

inputs8
conv2d_readvariableop_resource:@@-
biasadd_readvariableop_resource:@
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype0�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@*
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@g
IdentityIdentityBiasAdd:output:0^NoOp*
T0*/
_output_shapes
:���������@w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:���������@
 
_user_specified_nameinputs
�
�
#__inference_bn4_layer_call_fn_13250

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
>__inference_bn4_layer_call_and_return_conditional_losses_11192�
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
�
]
A__inference_lReLU3_layer_call_and_return_conditional_losses_13218

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
#__inference_bn1_layer_call_fn_12977

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
>__inference_bn1_layer_call_and_return_conditional_losses_11000�
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
�
�
'__inference_conv2D1_layer_call_fn_12954

inputs!
unknown: 
	unknown_0: 
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
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
GPU 2J 8� *K
fFRD
B__inference_conv2D1_layer_call_and_return_conditional_losses_11379w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:��������� `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������: : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
>__inference_bn3_layer_call_and_return_conditional_losses_11128

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
'__inference_conv2D2_layer_call_fn_13045

inputs!
unknown: @
	unknown_0:@
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
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
GPU 2J 8� *K
fFRD
B__inference_conv2D2_layer_call_and_return_conditional_losses_11411w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:���������@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:��������� : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:��������� 
 
_user_specified_nameinputs
�
B
&__inference_lReLU4_layer_call_fn_13304

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
A__inference_lReLU4_layer_call_and_return_conditional_losses_11495h
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
�
A__inference_logVar_layer_call_and_return_conditional_losses_11579

inputs1
matmul_readvariableop_resource:	�
-
biasadd_readvariableop_resource:

identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	�
*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:
*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������
_
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:���������
w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
�
>__inference_bn4_layer_call_and_return_conditional_losses_13299

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
�
?__inference_mean_layer_call_and_return_conditional_losses_13521

inputs1
matmul_readvariableop_resource:	�
-
biasadd_readvariableop_resource:

identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	�
*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:
*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������
_
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:���������
w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�

�
B__inference_conv2D3_layer_call_and_return_conditional_losses_13146

inputs8
conv2d_readvariableop_resource:@@-
biasadd_readvariableop_resource:@
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype0�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@*
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@g
IdentityIdentityBiasAdd:output:0^NoOp*
T0*/
_output_shapes
:���������@w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:���������@
 
_user_specified_nameinputs
�
�
'__inference_conv2D6_layer_call_fn_13409

inputs!
unknown:@ 
	unknown_0: 
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
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
GPU 2J 8� *K
fFRD
B__inference_conv2D6_layer_call_and_return_conditional_losses_11539w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:��������� `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������@: : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:���������@
 
_user_specified_nameinputs
�
�
#__inference_bn5_layer_call_fn_13354

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
>__inference_bn5_layer_call_and_return_conditional_losses_11274�
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

�
B__inference_conv2D2_layer_call_and_return_conditional_losses_13055

inputs8
conv2d_readvariableop_resource: @-
biasadd_readvariableop_resource:@
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: @*
dtype0�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@*
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@g
IdentityIdentityBiasAdd:output:0^NoOp*
T0*/
_output_shapes
:���������@w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:��������� : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:��������� 
 
_user_specified_nameinputs
�
B
&__inference_lReLU1_layer_call_fn_13031

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
A__inference_lReLU1_layer_call_and_return_conditional_losses_11399h
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
�
�
>__inference_bn6_layer_call_and_return_conditional_losses_13463

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
>__inference_bn2_layer_call_and_return_conditional_losses_11064

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
B__inference_conv2D1_layer_call_and_return_conditional_losses_12964

inputs8
conv2d_readvariableop_resource: -
biasadd_readvariableop_resource: 
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: *
dtype0�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:��������� *
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:��������� g
IdentityIdentityBiasAdd:output:0^NoOp*
T0*/
_output_shapes
:��������� w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
#__inference_bn1_layer_call_fn_12990

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
>__inference_bn1_layer_call_and_return_conditional_losses_11018�
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
�
�
>__inference_bn4_layer_call_and_return_conditional_losses_13281

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
B__inference_conv2D2_layer_call_and_return_conditional_losses_11411

inputs8
conv2d_readvariableop_resource: @-
biasadd_readvariableop_resource:@
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: @*
dtype0�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@*
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@g
IdentityIdentityBiasAdd:output:0^NoOp*
T0*/
_output_shapes
:���������@w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:��������� : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:��������� 
 
_user_specified_nameinputs
�
�	
'__inference_encoder_layer_call_fn_11903
input_layer!
unknown: 
	unknown_0: 
	unknown_1: 
	unknown_2: 
	unknown_3: 
	unknown_4: #
	unknown_5: @
	unknown_6:@
	unknown_7:@
	unknown_8:@
	unknown_9:@

unknown_10:@$

unknown_11:@@

unknown_12:@

unknown_13:@

unknown_14:@

unknown_15:@

unknown_16:@$

unknown_17:@@

unknown_18:@

unknown_19:@

unknown_20:@

unknown_21:@

unknown_22:@$

unknown_23:@@

unknown_24:@

unknown_25:@

unknown_26:@

unknown_27:@

unknown_28:@$

unknown_29:@ 

unknown_30: 

unknown_31: 

unknown_32: 

unknown_33: 

unknown_34: 

unknown_35:	�


unknown_36:


unknown_37:	�


unknown_38:

identity

identity_1��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinput_layerunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
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
unknown_32
unknown_33
unknown_34
unknown_35
unknown_36
unknown_37
unknown_38*4
Tin-
+2)*
Tout
2*
_collective_manager_ids
 *:
_output_shapes(
&:���������
:���������
*>
_read_only_resource_inputs 
	
 !"%&'(*-
config_proto

CPU

GPU 2J 8� *K
fFRD
B__inference_encoder_layer_call_and_return_conditional_losses_11818o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������
q

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*'
_output_shapes
:���������
`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*~
_input_shapesm
k:���������: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:\ X
/
_output_shapes
:���������
%
_user_specified_nameinput_layer
�
�
>__inference_bn5_layer_call_and_return_conditional_losses_11274

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
��
�
B__inference_encoder_layer_call_and_return_conditional_losses_12945

inputs@
&conv2d1_conv2d_readvariableop_resource: 5
'conv2d1_biasadd_readvariableop_resource: )
bn1_readvariableop_resource: +
bn1_readvariableop_1_resource: :
,bn1_fusedbatchnormv3_readvariableop_resource: <
.bn1_fusedbatchnormv3_readvariableop_1_resource: @
&conv2d2_conv2d_readvariableop_resource: @5
'conv2d2_biasadd_readvariableop_resource:@)
bn2_readvariableop_resource:@+
bn2_readvariableop_1_resource:@:
,bn2_fusedbatchnormv3_readvariableop_resource:@<
.bn2_fusedbatchnormv3_readvariableop_1_resource:@@
&conv2d3_conv2d_readvariableop_resource:@@5
'conv2d3_biasadd_readvariableop_resource:@)
bn3_readvariableop_resource:@+
bn3_readvariableop_1_resource:@:
,bn3_fusedbatchnormv3_readvariableop_resource:@<
.bn3_fusedbatchnormv3_readvariableop_1_resource:@@
&conv2d4_conv2d_readvariableop_resource:@@5
'conv2d4_biasadd_readvariableop_resource:@)
bn4_readvariableop_resource:@+
bn4_readvariableop_1_resource:@:
,bn4_fusedbatchnormv3_readvariableop_resource:@<
.bn4_fusedbatchnormv3_readvariableop_1_resource:@@
&conv2d5_conv2d_readvariableop_resource:@@5
'conv2d5_biasadd_readvariableop_resource:@)
bn5_readvariableop_resource:@+
bn5_readvariableop_1_resource:@:
,bn5_fusedbatchnormv3_readvariableop_resource:@<
.bn5_fusedbatchnormv3_readvariableop_1_resource:@@
&conv2d6_conv2d_readvariableop_resource:@ 5
'conv2d6_biasadd_readvariableop_resource: )
bn6_readvariableop_resource: +
bn6_readvariableop_1_resource: :
,bn6_fusedbatchnormv3_readvariableop_resource: <
.bn6_fusedbatchnormv3_readvariableop_1_resource: 8
%logvar_matmul_readvariableop_resource:	�
4
&logvar_biasadd_readvariableop_resource:
6
#mean_matmul_readvariableop_resource:	�
2
$mean_biasadd_readvariableop_resource:

identity

identity_1��#bn1/FusedBatchNormV3/ReadVariableOp�%bn1/FusedBatchNormV3/ReadVariableOp_1�bn1/ReadVariableOp�bn1/ReadVariableOp_1�#bn2/FusedBatchNormV3/ReadVariableOp�%bn2/FusedBatchNormV3/ReadVariableOp_1�bn2/ReadVariableOp�bn2/ReadVariableOp_1�#bn3/FusedBatchNormV3/ReadVariableOp�%bn3/FusedBatchNormV3/ReadVariableOp_1�bn3/ReadVariableOp�bn3/ReadVariableOp_1�#bn4/FusedBatchNormV3/ReadVariableOp�%bn4/FusedBatchNormV3/ReadVariableOp_1�bn4/ReadVariableOp�bn4/ReadVariableOp_1�#bn5/FusedBatchNormV3/ReadVariableOp�%bn5/FusedBatchNormV3/ReadVariableOp_1�bn5/ReadVariableOp�bn5/ReadVariableOp_1�#bn6/FusedBatchNormV3/ReadVariableOp�%bn6/FusedBatchNormV3/ReadVariableOp_1�bn6/ReadVariableOp�bn6/ReadVariableOp_1�conv2D1/BiasAdd/ReadVariableOp�conv2D1/Conv2D/ReadVariableOp�conv2D2/BiasAdd/ReadVariableOp�conv2D2/Conv2D/ReadVariableOp�conv2D3/BiasAdd/ReadVariableOp�conv2D3/Conv2D/ReadVariableOp�conv2D4/BiasAdd/ReadVariableOp�conv2D4/Conv2D/ReadVariableOp�conv2D5/BiasAdd/ReadVariableOp�conv2D5/Conv2D/ReadVariableOp�conv2D6/BiasAdd/ReadVariableOp�conv2D6/Conv2D/ReadVariableOp�logVar/BiasAdd/ReadVariableOp�logVar/MatMul/ReadVariableOp�mean/BiasAdd/ReadVariableOp�mean/MatMul/ReadVariableOp�
conv2D1/Conv2D/ReadVariableOpReadVariableOp&conv2d1_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype0�
conv2D1/Conv2DConv2Dinputs%conv2D1/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:��������� *
paddingSAME*
strides
�
conv2D1/BiasAdd/ReadVariableOpReadVariableOp'conv2d1_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0�
conv2D1/BiasAddBiasAddconv2D1/Conv2D:output:0&conv2D1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:��������� j
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
bn1/FusedBatchNormV3FusedBatchNormV3conv2D1/BiasAdd:output:0bn1/ReadVariableOp:value:0bn1/ReadVariableOp_1:value:0+bn1/FusedBatchNormV3/ReadVariableOp:value:0-bn1/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:��������� : : : : :*
epsilon%o�:*
is_training( x
lReLU1/LeakyRelu	LeakyRelubn1/FusedBatchNormV3:y:0*/
_output_shapes
:��������� *
alpha%���>�
conv2D2/Conv2D/ReadVariableOpReadVariableOp&conv2d2_conv2d_readvariableop_resource*&
_output_shapes
: @*
dtype0�
conv2D2/Conv2DConv2DlReLU1/LeakyRelu:activations:0%conv2D2/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@*
paddingSAME*
strides
�
conv2D2/BiasAdd/ReadVariableOpReadVariableOp'conv2d2_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0�
conv2D2/BiasAddBiasAddconv2D2/Conv2D:output:0&conv2D2/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@j
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
bn2/FusedBatchNormV3FusedBatchNormV3conv2D2/BiasAdd:output:0bn2/ReadVariableOp:value:0bn2/ReadVariableOp_1:value:0+bn2/FusedBatchNormV3/ReadVariableOp:value:0-bn2/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:���������@:@:@:@:@:*
epsilon%o�:*
is_training( x
lReLU2/LeakyRelu	LeakyRelubn2/FusedBatchNormV3:y:0*/
_output_shapes
:���������@*
alpha%���>�
conv2D3/Conv2D/ReadVariableOpReadVariableOp&conv2d3_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype0�
conv2D3/Conv2DConv2DlReLU2/LeakyRelu:activations:0%conv2D3/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@*
paddingSAME*
strides
�
conv2D3/BiasAdd/ReadVariableOpReadVariableOp'conv2d3_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0�
conv2D3/BiasAddBiasAddconv2D3/Conv2D:output:0&conv2D3/BiasAdd/ReadVariableOp:value:0*
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
bn3/FusedBatchNormV3FusedBatchNormV3conv2D3/BiasAdd:output:0bn3/ReadVariableOp:value:0bn3/ReadVariableOp_1:value:0+bn3/FusedBatchNormV3/ReadVariableOp:value:0-bn3/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:���������@:@:@:@:@:*
epsilon%o�:*
is_training( x
lReLU3/LeakyRelu	LeakyRelubn3/FusedBatchNormV3:y:0*/
_output_shapes
:���������@*
alpha%���>�
conv2D4/Conv2D/ReadVariableOpReadVariableOp&conv2d4_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype0�
conv2D4/Conv2DConv2DlReLU3/LeakyRelu:activations:0%conv2D4/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@*
paddingSAME*
strides
�
conv2D4/BiasAdd/ReadVariableOpReadVariableOp'conv2d4_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0�
conv2D4/BiasAddBiasAddconv2D4/Conv2D:output:0&conv2D4/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@j
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
bn4/FusedBatchNormV3FusedBatchNormV3conv2D4/BiasAdd:output:0bn4/ReadVariableOp:value:0bn4/ReadVariableOp_1:value:0+bn4/FusedBatchNormV3/ReadVariableOp:value:0-bn4/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:���������@:@:@:@:@:*
epsilon%o�:*
is_training( x
lReLU4/LeakyRelu	LeakyRelubn4/FusedBatchNormV3:y:0*/
_output_shapes
:���������@*
alpha%���>�
conv2D5/Conv2D/ReadVariableOpReadVariableOp&conv2d5_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype0�
conv2D5/Conv2DConv2DlReLU4/LeakyRelu:activations:0%conv2D5/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@*
paddingSAME*
strides
�
conv2D5/BiasAdd/ReadVariableOpReadVariableOp'conv2d5_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0�
conv2D5/BiasAddBiasAddconv2D5/Conv2D:output:0&conv2D5/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@j
bn5/ReadVariableOpReadVariableOpbn5_readvariableop_resource*
_output_shapes
:@*
dtype0n
bn5/ReadVariableOp_1ReadVariableOpbn5_readvariableop_1_resource*
_output_shapes
:@*
dtype0�
#bn5/FusedBatchNormV3/ReadVariableOpReadVariableOp,bn5_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype0�
%bn5/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp.bn5_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype0�
bn5/FusedBatchNormV3FusedBatchNormV3conv2D5/BiasAdd:output:0bn5/ReadVariableOp:value:0bn5/ReadVariableOp_1:value:0+bn5/FusedBatchNormV3/ReadVariableOp:value:0-bn5/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:���������@:@:@:@:@:*
epsilon%o�:*
is_training( x
lReLU5/LeakyRelu	LeakyRelubn5/FusedBatchNormV3:y:0*/
_output_shapes
:���������@*
alpha%���>�
conv2D6/Conv2D/ReadVariableOpReadVariableOp&conv2d6_conv2d_readvariableop_resource*&
_output_shapes
:@ *
dtype0�
conv2D6/Conv2DConv2DlReLU5/LeakyRelu:activations:0%conv2D6/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:��������� *
paddingSAME*
strides
�
conv2D6/BiasAdd/ReadVariableOpReadVariableOp'conv2d6_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0�
conv2D6/BiasAddBiasAddconv2D6/Conv2D:output:0&conv2D6/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:��������� j
bn6/ReadVariableOpReadVariableOpbn6_readvariableop_resource*
_output_shapes
: *
dtype0n
bn6/ReadVariableOp_1ReadVariableOpbn6_readvariableop_1_resource*
_output_shapes
: *
dtype0�
#bn6/FusedBatchNormV3/ReadVariableOpReadVariableOp,bn6_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype0�
%bn6/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp.bn6_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype0�
bn6/FusedBatchNormV3FusedBatchNormV3conv2D6/BiasAdd:output:0bn6/ReadVariableOp:value:0bn6/ReadVariableOp_1:value:0+bn6/FusedBatchNormV3/ReadVariableOp:value:0-bn6/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:��������� : : : : :*
epsilon%o�:*
is_training( x
lReLU6/LeakyRelu	LeakyRelubn6/FusedBatchNormV3:y:0*/
_output_shapes
:��������� *
alpha%���>^
flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"����   �
flatten/ReshapeReshapelReLU6/LeakyRelu:activations:0flatten/Const:output:0*
T0*(
_output_shapes
:�����������
logVar/MatMul/ReadVariableOpReadVariableOp%logvar_matmul_readvariableop_resource*
_output_shapes
:	�
*
dtype0�
logVar/MatMulMatMulflatten/Reshape:output:0$logVar/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������
�
logVar/BiasAdd/ReadVariableOpReadVariableOp&logvar_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype0�
logVar/BiasAddBiasAddlogVar/MatMul:product:0%logVar/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������

mean/MatMul/ReadVariableOpReadVariableOp#mean_matmul_readvariableop_resource*
_output_shapes
:	�
*
dtype0�
mean/MatMulMatMulflatten/Reshape:output:0"mean/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������
|
mean/BiasAdd/ReadVariableOpReadVariableOp$mean_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype0�
mean/BiasAddBiasAddmean/MatMul:product:0#mean/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������
d
IdentityIdentitymean/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:���������
h

Identity_1IdentitylogVar/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:���������
�

NoOpNoOp$^bn1/FusedBatchNormV3/ReadVariableOp&^bn1/FusedBatchNormV3/ReadVariableOp_1^bn1/ReadVariableOp^bn1/ReadVariableOp_1$^bn2/FusedBatchNormV3/ReadVariableOp&^bn2/FusedBatchNormV3/ReadVariableOp_1^bn2/ReadVariableOp^bn2/ReadVariableOp_1$^bn3/FusedBatchNormV3/ReadVariableOp&^bn3/FusedBatchNormV3/ReadVariableOp_1^bn3/ReadVariableOp^bn3/ReadVariableOp_1$^bn4/FusedBatchNormV3/ReadVariableOp&^bn4/FusedBatchNormV3/ReadVariableOp_1^bn4/ReadVariableOp^bn4/ReadVariableOp_1$^bn5/FusedBatchNormV3/ReadVariableOp&^bn5/FusedBatchNormV3/ReadVariableOp_1^bn5/ReadVariableOp^bn5/ReadVariableOp_1$^bn6/FusedBatchNormV3/ReadVariableOp&^bn6/FusedBatchNormV3/ReadVariableOp_1^bn6/ReadVariableOp^bn6/ReadVariableOp_1^conv2D1/BiasAdd/ReadVariableOp^conv2D1/Conv2D/ReadVariableOp^conv2D2/BiasAdd/ReadVariableOp^conv2D2/Conv2D/ReadVariableOp^conv2D3/BiasAdd/ReadVariableOp^conv2D3/Conv2D/ReadVariableOp^conv2D4/BiasAdd/ReadVariableOp^conv2D4/Conv2D/ReadVariableOp^conv2D5/BiasAdd/ReadVariableOp^conv2D5/Conv2D/ReadVariableOp^conv2D6/BiasAdd/ReadVariableOp^conv2D6/Conv2D/ReadVariableOp^logVar/BiasAdd/ReadVariableOp^logVar/MatMul/ReadVariableOp^mean/BiasAdd/ReadVariableOp^mean/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*~
_input_shapesm
k:���������: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2J
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
bn5/ReadVariableOp_1bn5/ReadVariableOp_12J
#bn6/FusedBatchNormV3/ReadVariableOp#bn6/FusedBatchNormV3/ReadVariableOp2N
%bn6/FusedBatchNormV3/ReadVariableOp_1%bn6/FusedBatchNormV3/ReadVariableOp_12(
bn6/ReadVariableOpbn6/ReadVariableOp2,
bn6/ReadVariableOp_1bn6/ReadVariableOp_12@
conv2D1/BiasAdd/ReadVariableOpconv2D1/BiasAdd/ReadVariableOp2>
conv2D1/Conv2D/ReadVariableOpconv2D1/Conv2D/ReadVariableOp2@
conv2D2/BiasAdd/ReadVariableOpconv2D2/BiasAdd/ReadVariableOp2>
conv2D2/Conv2D/ReadVariableOpconv2D2/Conv2D/ReadVariableOp2@
conv2D3/BiasAdd/ReadVariableOpconv2D3/BiasAdd/ReadVariableOp2>
conv2D3/Conv2D/ReadVariableOpconv2D3/Conv2D/ReadVariableOp2@
conv2D4/BiasAdd/ReadVariableOpconv2D4/BiasAdd/ReadVariableOp2>
conv2D4/Conv2D/ReadVariableOpconv2D4/Conv2D/ReadVariableOp2@
conv2D5/BiasAdd/ReadVariableOpconv2D5/BiasAdd/ReadVariableOp2>
conv2D5/Conv2D/ReadVariableOpconv2D5/Conv2D/ReadVariableOp2@
conv2D6/BiasAdd/ReadVariableOpconv2D6/BiasAdd/ReadVariableOp2>
conv2D6/Conv2D/ReadVariableOpconv2D6/Conv2D/ReadVariableOp2>
logVar/BiasAdd/ReadVariableOplogVar/BiasAdd/ReadVariableOp2<
logVar/MatMul/ReadVariableOplogVar/MatMul/ReadVariableOp2:
mean/BiasAdd/ReadVariableOpmean/BiasAdd/ReadVariableOp28
mean/MatMul/ReadVariableOpmean/MatMul/ReadVariableOp:W S
/
_output_shapes
:���������
 
_user_specified_nameinputs
�
B
&__inference_lReLU2_layer_call_fn_13122

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
A__inference_lReLU2_layer_call_and_return_conditional_losses_11431h
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
#__inference_bn2_layer_call_fn_13081

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
>__inference_bn2_layer_call_and_return_conditional_losses_11082�
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

�
B__inference_conv2D3_layer_call_and_return_conditional_losses_11443

inputs8
conv2d_readvariableop_resource:@@-
biasadd_readvariableop_resource:@
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype0�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@*
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@g
IdentityIdentityBiasAdd:output:0^NoOp*
T0*/
_output_shapes
:���������@w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:���������@
 
_user_specified_nameinputs
�[
�
B__inference_encoder_layer_call_and_return_conditional_losses_11709
input_layer'
conv2d1_11606: 
conv2d1_11608: 
	bn1_11611: 
	bn1_11613: 
	bn1_11615: 
	bn1_11617: '
conv2d2_11621: @
conv2d2_11623:@
	bn2_11626:@
	bn2_11628:@
	bn2_11630:@
	bn2_11632:@'
conv2d3_11636:@@
conv2d3_11638:@
	bn3_11641:@
	bn3_11643:@
	bn3_11645:@
	bn3_11647:@'
conv2d4_11651:@@
conv2d4_11653:@
	bn4_11656:@
	bn4_11658:@
	bn4_11660:@
	bn4_11662:@'
conv2d5_11666:@@
conv2d5_11668:@
	bn5_11671:@
	bn5_11673:@
	bn5_11675:@
	bn5_11677:@'
conv2d6_11681:@ 
conv2d6_11683: 
	bn6_11686: 
	bn6_11688: 
	bn6_11690: 
	bn6_11692: 
logvar_11697:	�

logvar_11699:


mean_11702:	�


mean_11704:

identity

identity_1��bn1/StatefulPartitionedCall�bn2/StatefulPartitionedCall�bn3/StatefulPartitionedCall�bn4/StatefulPartitionedCall�bn5/StatefulPartitionedCall�bn6/StatefulPartitionedCall�conv2D1/StatefulPartitionedCall�conv2D2/StatefulPartitionedCall�conv2D3/StatefulPartitionedCall�conv2D4/StatefulPartitionedCall�conv2D5/StatefulPartitionedCall�conv2D6/StatefulPartitionedCall�logVar/StatefulPartitionedCall�mean/StatefulPartitionedCall�
conv2D1/StatefulPartitionedCallStatefulPartitionedCallinput_layerconv2d1_11606conv2d1_11608*
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
GPU 2J 8� *K
fFRD
B__inference_conv2D1_layer_call_and_return_conditional_losses_11379�
bn1/StatefulPartitionedCallStatefulPartitionedCall(conv2D1/StatefulPartitionedCall:output:0	bn1_11611	bn1_11613	bn1_11615	bn1_11617*
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
>__inference_bn1_layer_call_and_return_conditional_losses_11018�
lReLU1/PartitionedCallPartitionedCall$bn1/StatefulPartitionedCall:output:0*
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
A__inference_lReLU1_layer_call_and_return_conditional_losses_11399�
conv2D2/StatefulPartitionedCallStatefulPartitionedCalllReLU1/PartitionedCall:output:0conv2d2_11621conv2d2_11623*
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
GPU 2J 8� *K
fFRD
B__inference_conv2D2_layer_call_and_return_conditional_losses_11411�
bn2/StatefulPartitionedCallStatefulPartitionedCall(conv2D2/StatefulPartitionedCall:output:0	bn2_11626	bn2_11628	bn2_11630	bn2_11632*
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
>__inference_bn2_layer_call_and_return_conditional_losses_11082�
lReLU2/PartitionedCallPartitionedCall$bn2/StatefulPartitionedCall:output:0*
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
A__inference_lReLU2_layer_call_and_return_conditional_losses_11431�
conv2D3/StatefulPartitionedCallStatefulPartitionedCalllReLU2/PartitionedCall:output:0conv2d3_11636conv2d3_11638*
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
GPU 2J 8� *K
fFRD
B__inference_conv2D3_layer_call_and_return_conditional_losses_11443�
bn3/StatefulPartitionedCallStatefulPartitionedCall(conv2D3/StatefulPartitionedCall:output:0	bn3_11641	bn3_11643	bn3_11645	bn3_11647*
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
>__inference_bn3_layer_call_and_return_conditional_losses_11146�
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
A__inference_lReLU3_layer_call_and_return_conditional_losses_11463�
conv2D4/StatefulPartitionedCallStatefulPartitionedCalllReLU3/PartitionedCall:output:0conv2d4_11651conv2d4_11653*
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
GPU 2J 8� *K
fFRD
B__inference_conv2D4_layer_call_and_return_conditional_losses_11475�
bn4/StatefulPartitionedCallStatefulPartitionedCall(conv2D4/StatefulPartitionedCall:output:0	bn4_11656	bn4_11658	bn4_11660	bn4_11662*
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
>__inference_bn4_layer_call_and_return_conditional_losses_11210�
lReLU4/PartitionedCallPartitionedCall$bn4/StatefulPartitionedCall:output:0*
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
A__inference_lReLU4_layer_call_and_return_conditional_losses_11495�
conv2D5/StatefulPartitionedCallStatefulPartitionedCalllReLU4/PartitionedCall:output:0conv2d5_11666conv2d5_11668*
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
GPU 2J 8� *K
fFRD
B__inference_conv2D5_layer_call_and_return_conditional_losses_11507�
bn5/StatefulPartitionedCallStatefulPartitionedCall(conv2D5/StatefulPartitionedCall:output:0	bn5_11671	bn5_11673	bn5_11675	bn5_11677*
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
>__inference_bn5_layer_call_and_return_conditional_losses_11274�
lReLU5/PartitionedCallPartitionedCall$bn5/StatefulPartitionedCall:output:0*
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
A__inference_lReLU5_layer_call_and_return_conditional_losses_11527�
conv2D6/StatefulPartitionedCallStatefulPartitionedCalllReLU5/PartitionedCall:output:0conv2d6_11681conv2d6_11683*
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
GPU 2J 8� *K
fFRD
B__inference_conv2D6_layer_call_and_return_conditional_losses_11539�
bn6/StatefulPartitionedCallStatefulPartitionedCall(conv2D6/StatefulPartitionedCall:output:0	bn6_11686	bn6_11688	bn6_11690	bn6_11692*
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
>__inference_bn6_layer_call_and_return_conditional_losses_11338�
lReLU6/PartitionedCallPartitionedCall$bn6/StatefulPartitionedCall:output:0*
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
A__inference_lReLU6_layer_call_and_return_conditional_losses_11559�
flatten/PartitionedCallPartitionedCalllReLU6/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *K
fFRD
B__inference_flatten_layer_call_and_return_conditional_losses_11567�
logVar/StatefulPartitionedCallStatefulPartitionedCall flatten/PartitionedCall:output:0logvar_11697logvar_11699*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������
*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *J
fERC
A__inference_logVar_layer_call_and_return_conditional_losses_11579�
mean/StatefulPartitionedCallStatefulPartitionedCall flatten/PartitionedCall:output:0
mean_11702
mean_11704*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������
*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *H
fCRA
?__inference_mean_layer_call_and_return_conditional_losses_11595t
IdentityIdentity%mean/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������
x

Identity_1Identity'logVar/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������
�
NoOpNoOp^bn1/StatefulPartitionedCall^bn2/StatefulPartitionedCall^bn3/StatefulPartitionedCall^bn4/StatefulPartitionedCall^bn5/StatefulPartitionedCall^bn6/StatefulPartitionedCall ^conv2D1/StatefulPartitionedCall ^conv2D2/StatefulPartitionedCall ^conv2D3/StatefulPartitionedCall ^conv2D4/StatefulPartitionedCall ^conv2D5/StatefulPartitionedCall ^conv2D6/StatefulPartitionedCall^logVar/StatefulPartitionedCall^mean/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*~
_input_shapesm
k:���������: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2:
bn1/StatefulPartitionedCallbn1/StatefulPartitionedCall2:
bn2/StatefulPartitionedCallbn2/StatefulPartitionedCall2:
bn3/StatefulPartitionedCallbn3/StatefulPartitionedCall2:
bn4/StatefulPartitionedCallbn4/StatefulPartitionedCall2:
bn5/StatefulPartitionedCallbn5/StatefulPartitionedCall2:
bn6/StatefulPartitionedCallbn6/StatefulPartitionedCall2B
conv2D1/StatefulPartitionedCallconv2D1/StatefulPartitionedCall2B
conv2D2/StatefulPartitionedCallconv2D2/StatefulPartitionedCall2B
conv2D3/StatefulPartitionedCallconv2D3/StatefulPartitionedCall2B
conv2D4/StatefulPartitionedCallconv2D4/StatefulPartitionedCall2B
conv2D5/StatefulPartitionedCallconv2D5/StatefulPartitionedCall2B
conv2D6/StatefulPartitionedCallconv2D6/StatefulPartitionedCall2@
logVar/StatefulPartitionedCalllogVar/StatefulPartitionedCall2<
mean/StatefulPartitionedCallmean/StatefulPartitionedCall:\ X
/
_output_shapes
:���������
%
_user_specified_nameinput_layer
�
�
#__inference_bn5_layer_call_fn_13341

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
>__inference_bn5_layer_call_and_return_conditional_losses_11256�
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
>__inference_bn3_layer_call_and_return_conditional_losses_13190

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
#__inference_bn6_layer_call_fn_13432

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
>__inference_bn6_layer_call_and_return_conditional_losses_11320�
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
�
�	
'__inference_encoder_layer_call_fn_12096
input_layer!
unknown: 
	unknown_0: 
	unknown_1: 
	unknown_2: 
	unknown_3: 
	unknown_4: #
	unknown_5: @
	unknown_6:@
	unknown_7:@
	unknown_8:@
	unknown_9:@

unknown_10:@$

unknown_11:@@

unknown_12:@

unknown_13:@

unknown_14:@

unknown_15:@

unknown_16:@$

unknown_17:@@

unknown_18:@

unknown_19:@

unknown_20:@

unknown_21:@

unknown_22:@$

unknown_23:@@

unknown_24:@

unknown_25:@

unknown_26:@

unknown_27:@

unknown_28:@$

unknown_29:@ 

unknown_30: 

unknown_31: 

unknown_32: 

unknown_33: 

unknown_34: 

unknown_35:	�


unknown_36:


unknown_37:	�


unknown_38:

identity

identity_1��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinput_layerunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
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
unknown_32
unknown_33
unknown_34
unknown_35
unknown_36
unknown_37
unknown_38*4
Tin-
+2)*
Tout
2*
_collective_manager_ids
 *:
_output_shapes(
&:���������
:���������
*J
_read_only_resource_inputs,
*(	
 !"#$%&'(*-
config_proto

CPU

GPU 2J 8� *K
fFRD
B__inference_encoder_layer_call_and_return_conditional_losses_12011o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������
q

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*'
_output_shapes
:���������
`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*~
_input_shapesm
k:���������: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:\ X
/
_output_shapes
:���������
%
_user_specified_nameinput_layer
�
�
>__inference_bn6_layer_call_and_return_conditional_losses_11338

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
A__inference_lReLU4_layer_call_and_return_conditional_losses_11495

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
�
�
>__inference_bn4_layer_call_and_return_conditional_losses_11192

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
#__inference_bn2_layer_call_fn_13068

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
>__inference_bn2_layer_call_and_return_conditional_losses_11064�
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
#__inference_bn3_layer_call_fn_13172

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
>__inference_bn3_layer_call_and_return_conditional_losses_11146�
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
�
�	
'__inference_encoder_layer_call_fn_12655

inputs!
unknown: 
	unknown_0: 
	unknown_1: 
	unknown_2: 
	unknown_3: 
	unknown_4: #
	unknown_5: @
	unknown_6:@
	unknown_7:@
	unknown_8:@
	unknown_9:@

unknown_10:@$

unknown_11:@@

unknown_12:@

unknown_13:@

unknown_14:@

unknown_15:@

unknown_16:@$

unknown_17:@@

unknown_18:@

unknown_19:@

unknown_20:@

unknown_21:@

unknown_22:@$

unknown_23:@@

unknown_24:@

unknown_25:@

unknown_26:@

unknown_27:@

unknown_28:@$

unknown_29:@ 

unknown_30: 

unknown_31: 

unknown_32: 

unknown_33: 

unknown_34: 

unknown_35:	�


unknown_36:


unknown_37:	�


unknown_38:

identity

identity_1��StatefulPartitionedCall�
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
unknown_32
unknown_33
unknown_34
unknown_35
unknown_36
unknown_37
unknown_38*4
Tin-
+2)*
Tout
2*
_collective_manager_ids
 *:
_output_shapes(
&:���������
:���������
*J
_read_only_resource_inputs,
*(	
 !"#$%&'(*-
config_proto

CPU

GPU 2J 8� *K
fFRD
B__inference_encoder_layer_call_and_return_conditional_losses_12011o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������
q

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*'
_output_shapes
:���������
`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*~
_input_shapesm
k:���������: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:���������
 
_user_specified_nameinputs
�
]
A__inference_lReLU3_layer_call_and_return_conditional_losses_11463

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
��
�
!__inference__traced_restore_13934
file_prefix9
assignvariableop_conv2d1_kernel: -
assignvariableop_1_conv2d1_bias: *
assignvariableop_2_bn1_gamma: )
assignvariableop_3_bn1_beta: 0
"assignvariableop_4_bn1_moving_mean: 4
&assignvariableop_5_bn1_moving_variance: ;
!assignvariableop_6_conv2d2_kernel: @-
assignvariableop_7_conv2d2_bias:@*
assignvariableop_8_bn2_gamma:@)
assignvariableop_9_bn2_beta:@1
#assignvariableop_10_bn2_moving_mean:@5
'assignvariableop_11_bn2_moving_variance:@<
"assignvariableop_12_conv2d3_kernel:@@.
 assignvariableop_13_conv2d3_bias:@+
assignvariableop_14_bn3_gamma:@*
assignvariableop_15_bn3_beta:@1
#assignvariableop_16_bn3_moving_mean:@5
'assignvariableop_17_bn3_moving_variance:@<
"assignvariableop_18_conv2d4_kernel:@@.
 assignvariableop_19_conv2d4_bias:@+
assignvariableop_20_bn4_gamma:@*
assignvariableop_21_bn4_beta:@1
#assignvariableop_22_bn4_moving_mean:@5
'assignvariableop_23_bn4_moving_variance:@<
"assignvariableop_24_conv2d5_kernel:@@.
 assignvariableop_25_conv2d5_bias:@+
assignvariableop_26_bn5_gamma:@*
assignvariableop_27_bn5_beta:@1
#assignvariableop_28_bn5_moving_mean:@5
'assignvariableop_29_bn5_moving_variance:@<
"assignvariableop_30_conv2d6_kernel:@ .
 assignvariableop_31_conv2d6_bias: +
assignvariableop_32_bn6_gamma: *
assignvariableop_33_bn6_beta: 1
#assignvariableop_34_bn6_moving_mean: 5
'assignvariableop_35_bn6_moving_variance: 2
assignvariableop_36_mean_kernel:	�
+
assignvariableop_37_mean_bias:
4
!assignvariableop_38_logvar_kernel:	�
-
assignvariableop_39_logvar_bias:

identity_41��AssignVariableOp�AssignVariableOp_1�AssignVariableOp_10�AssignVariableOp_11�AssignVariableOp_12�AssignVariableOp_13�AssignVariableOp_14�AssignVariableOp_15�AssignVariableOp_16�AssignVariableOp_17�AssignVariableOp_18�AssignVariableOp_19�AssignVariableOp_2�AssignVariableOp_20�AssignVariableOp_21�AssignVariableOp_22�AssignVariableOp_23�AssignVariableOp_24�AssignVariableOp_25�AssignVariableOp_26�AssignVariableOp_27�AssignVariableOp_28�AssignVariableOp_29�AssignVariableOp_3�AssignVariableOp_30�AssignVariableOp_31�AssignVariableOp_32�AssignVariableOp_33�AssignVariableOp_34�AssignVariableOp_35�AssignVariableOp_36�AssignVariableOp_37�AssignVariableOp_38�AssignVariableOp_39�AssignVariableOp_4�AssignVariableOp_5�AssignVariableOp_6�AssignVariableOp_7�AssignVariableOp_8�AssignVariableOp_9�
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:)*
dtype0*�
value�B�)B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-1/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-1/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-1/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-3/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-3/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-3/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-5/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-5/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-5/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-7/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-7/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-7/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-7/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-8/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-8/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-9/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-9/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-9/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-9/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-10/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-10/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-11/gamma/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-11/beta/.ATTRIBUTES/VARIABLE_VALUEB<layer_with_weights-11/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-11/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-12/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-12/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-13/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-13/bias/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH�
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:)*
dtype0*e
value\BZ)B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B �
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*�
_output_shapes�
�:::::::::::::::::::::::::::::::::::::::::*7
dtypes-
+2)[
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOpAssignVariableOpassignvariableop_conv2d1_kernelIdentity:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_1AssignVariableOpassignvariableop_1_conv2d1_biasIdentity_1:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_2AssignVariableOpassignvariableop_2_bn1_gammaIdentity_2:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_3AssignVariableOpassignvariableop_3_bn1_betaIdentity_3:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_4AssignVariableOp"assignvariableop_4_bn1_moving_meanIdentity_4:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_5AssignVariableOp&assignvariableop_5_bn1_moving_varianceIdentity_5:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_6AssignVariableOp!assignvariableop_6_conv2d2_kernelIdentity_6:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_7AssignVariableOpassignvariableop_7_conv2d2_biasIdentity_7:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_8AssignVariableOpassignvariableop_8_bn2_gammaIdentity_8:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_9AssignVariableOpassignvariableop_9_bn2_betaIdentity_9:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_10AssignVariableOp#assignvariableop_10_bn2_moving_meanIdentity_10:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_11AssignVariableOp'assignvariableop_11_bn2_moving_varianceIdentity_11:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_12AssignVariableOp"assignvariableop_12_conv2d3_kernelIdentity_12:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_13AssignVariableOp assignvariableop_13_conv2d3_biasIdentity_13:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_14AssignVariableOpassignvariableop_14_bn3_gammaIdentity_14:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_15AssignVariableOpassignvariableop_15_bn3_betaIdentity_15:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_16AssignVariableOp#assignvariableop_16_bn3_moving_meanIdentity_16:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_17AssignVariableOp'assignvariableop_17_bn3_moving_varianceIdentity_17:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_18AssignVariableOp"assignvariableop_18_conv2d4_kernelIdentity_18:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_19AssignVariableOp assignvariableop_19_conv2d4_biasIdentity_19:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_20AssignVariableOpassignvariableop_20_bn4_gammaIdentity_20:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_21AssignVariableOpassignvariableop_21_bn4_betaIdentity_21:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_22AssignVariableOp#assignvariableop_22_bn4_moving_meanIdentity_22:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_23AssignVariableOp'assignvariableop_23_bn4_moving_varianceIdentity_23:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_24AssignVariableOp"assignvariableop_24_conv2d5_kernelIdentity_24:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_25AssignVariableOp assignvariableop_25_conv2d5_biasIdentity_25:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_26AssignVariableOpassignvariableop_26_bn5_gammaIdentity_26:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_27AssignVariableOpassignvariableop_27_bn5_betaIdentity_27:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_28AssignVariableOp#assignvariableop_28_bn5_moving_meanIdentity_28:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_29AssignVariableOp'assignvariableop_29_bn5_moving_varianceIdentity_29:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_30AssignVariableOp"assignvariableop_30_conv2d6_kernelIdentity_30:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_31AssignVariableOp assignvariableop_31_conv2d6_biasIdentity_31:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_32AssignVariableOpassignvariableop_32_bn6_gammaIdentity_32:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_33IdentityRestoreV2:tensors:33"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_33AssignVariableOpassignvariableop_33_bn6_betaIdentity_33:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_34IdentityRestoreV2:tensors:34"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_34AssignVariableOp#assignvariableop_34_bn6_moving_meanIdentity_34:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_35IdentityRestoreV2:tensors:35"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_35AssignVariableOp'assignvariableop_35_bn6_moving_varianceIdentity_35:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_36IdentityRestoreV2:tensors:36"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_36AssignVariableOpassignvariableop_36_mean_kernelIdentity_36:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_37IdentityRestoreV2:tensors:37"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_37AssignVariableOpassignvariableop_37_mean_biasIdentity_37:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_38IdentityRestoreV2:tensors:38"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_38AssignVariableOp!assignvariableop_38_logvar_kernelIdentity_38:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_39IdentityRestoreV2:tensors:39"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_39AssignVariableOpassignvariableop_39_logvar_biasIdentity_39:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0Y
NoOpNoOp"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 �
Identity_40Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: W
Identity_41IdentityIdentity_40:output:0^NoOp_1*
T0*
_output_shapes
: �
NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*"
_acd_function_control_output(*
_output_shapes
 "#
identity_41Identity_41:output:0*e
_input_shapesT
R: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2$
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
AssignVariableOp_33AssignVariableOp_332*
AssignVariableOp_34AssignVariableOp_342*
AssignVariableOp_35AssignVariableOp_352*
AssignVariableOp_36AssignVariableOp_362*
AssignVariableOp_37AssignVariableOp_372*
AssignVariableOp_38AssignVariableOp_382*
AssignVariableOp_39AssignVariableOp_392(
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
>__inference_bn1_layer_call_and_return_conditional_losses_11018

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
�
�
$__inference_mean_layer_call_fn_13511

inputs
unknown:	�

	unknown_0:

identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������
*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *H
fCRA
?__inference_mean_layer_call_and_return_conditional_losses_11595o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������
`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
�
>__inference_bn1_layer_call_and_return_conditional_losses_11000

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
#__inference_bn6_layer_call_fn_13445

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
>__inference_bn6_layer_call_and_return_conditional_losses_11338�
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
A__inference_lReLU5_layer_call_and_return_conditional_losses_11527

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

�
B__inference_conv2D4_layer_call_and_return_conditional_losses_11475

inputs8
conv2d_readvariableop_resource:@@-
biasadd_readvariableop_resource:@
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype0�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@*
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@g
IdentityIdentityBiasAdd:output:0^NoOp*
T0*/
_output_shapes
:���������@w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:���������@
 
_user_specified_nameinputs
�
�
>__inference_bn3_layer_call_and_return_conditional_losses_11146

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
�[
�
B__inference_encoder_layer_call_and_return_conditional_losses_12011

inputs'
conv2d1_11908: 
conv2d1_11910: 
	bn1_11913: 
	bn1_11915: 
	bn1_11917: 
	bn1_11919: '
conv2d2_11923: @
conv2d2_11925:@
	bn2_11928:@
	bn2_11930:@
	bn2_11932:@
	bn2_11934:@'
conv2d3_11938:@@
conv2d3_11940:@
	bn3_11943:@
	bn3_11945:@
	bn3_11947:@
	bn3_11949:@'
conv2d4_11953:@@
conv2d4_11955:@
	bn4_11958:@
	bn4_11960:@
	bn4_11962:@
	bn4_11964:@'
conv2d5_11968:@@
conv2d5_11970:@
	bn5_11973:@
	bn5_11975:@
	bn5_11977:@
	bn5_11979:@'
conv2d6_11983:@ 
conv2d6_11985: 
	bn6_11988: 
	bn6_11990: 
	bn6_11992: 
	bn6_11994: 
logvar_11999:	�

logvar_12001:


mean_12004:	�


mean_12006:

identity

identity_1��bn1/StatefulPartitionedCall�bn2/StatefulPartitionedCall�bn3/StatefulPartitionedCall�bn4/StatefulPartitionedCall�bn5/StatefulPartitionedCall�bn6/StatefulPartitionedCall�conv2D1/StatefulPartitionedCall�conv2D2/StatefulPartitionedCall�conv2D3/StatefulPartitionedCall�conv2D4/StatefulPartitionedCall�conv2D5/StatefulPartitionedCall�conv2D6/StatefulPartitionedCall�logVar/StatefulPartitionedCall�mean/StatefulPartitionedCall�
conv2D1/StatefulPartitionedCallStatefulPartitionedCallinputsconv2d1_11908conv2d1_11910*
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
GPU 2J 8� *K
fFRD
B__inference_conv2D1_layer_call_and_return_conditional_losses_11379�
bn1/StatefulPartitionedCallStatefulPartitionedCall(conv2D1/StatefulPartitionedCall:output:0	bn1_11913	bn1_11915	bn1_11917	bn1_11919*
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
>__inference_bn1_layer_call_and_return_conditional_losses_11018�
lReLU1/PartitionedCallPartitionedCall$bn1/StatefulPartitionedCall:output:0*
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
A__inference_lReLU1_layer_call_and_return_conditional_losses_11399�
conv2D2/StatefulPartitionedCallStatefulPartitionedCalllReLU1/PartitionedCall:output:0conv2d2_11923conv2d2_11925*
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
GPU 2J 8� *K
fFRD
B__inference_conv2D2_layer_call_and_return_conditional_losses_11411�
bn2/StatefulPartitionedCallStatefulPartitionedCall(conv2D2/StatefulPartitionedCall:output:0	bn2_11928	bn2_11930	bn2_11932	bn2_11934*
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
>__inference_bn2_layer_call_and_return_conditional_losses_11082�
lReLU2/PartitionedCallPartitionedCall$bn2/StatefulPartitionedCall:output:0*
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
A__inference_lReLU2_layer_call_and_return_conditional_losses_11431�
conv2D3/StatefulPartitionedCallStatefulPartitionedCalllReLU2/PartitionedCall:output:0conv2d3_11938conv2d3_11940*
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
GPU 2J 8� *K
fFRD
B__inference_conv2D3_layer_call_and_return_conditional_losses_11443�
bn3/StatefulPartitionedCallStatefulPartitionedCall(conv2D3/StatefulPartitionedCall:output:0	bn3_11943	bn3_11945	bn3_11947	bn3_11949*
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
>__inference_bn3_layer_call_and_return_conditional_losses_11146�
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
A__inference_lReLU3_layer_call_and_return_conditional_losses_11463�
conv2D4/StatefulPartitionedCallStatefulPartitionedCalllReLU3/PartitionedCall:output:0conv2d4_11953conv2d4_11955*
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
GPU 2J 8� *K
fFRD
B__inference_conv2D4_layer_call_and_return_conditional_losses_11475�
bn4/StatefulPartitionedCallStatefulPartitionedCall(conv2D4/StatefulPartitionedCall:output:0	bn4_11958	bn4_11960	bn4_11962	bn4_11964*
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
>__inference_bn4_layer_call_and_return_conditional_losses_11210�
lReLU4/PartitionedCallPartitionedCall$bn4/StatefulPartitionedCall:output:0*
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
A__inference_lReLU4_layer_call_and_return_conditional_losses_11495�
conv2D5/StatefulPartitionedCallStatefulPartitionedCalllReLU4/PartitionedCall:output:0conv2d5_11968conv2d5_11970*
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
GPU 2J 8� *K
fFRD
B__inference_conv2D5_layer_call_and_return_conditional_losses_11507�
bn5/StatefulPartitionedCallStatefulPartitionedCall(conv2D5/StatefulPartitionedCall:output:0	bn5_11973	bn5_11975	bn5_11977	bn5_11979*
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
>__inference_bn5_layer_call_and_return_conditional_losses_11274�
lReLU5/PartitionedCallPartitionedCall$bn5/StatefulPartitionedCall:output:0*
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
A__inference_lReLU5_layer_call_and_return_conditional_losses_11527�
conv2D6/StatefulPartitionedCallStatefulPartitionedCalllReLU5/PartitionedCall:output:0conv2d6_11983conv2d6_11985*
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
GPU 2J 8� *K
fFRD
B__inference_conv2D6_layer_call_and_return_conditional_losses_11539�
bn6/StatefulPartitionedCallStatefulPartitionedCall(conv2D6/StatefulPartitionedCall:output:0	bn6_11988	bn6_11990	bn6_11992	bn6_11994*
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
>__inference_bn6_layer_call_and_return_conditional_losses_11338�
lReLU6/PartitionedCallPartitionedCall$bn6/StatefulPartitionedCall:output:0*
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
A__inference_lReLU6_layer_call_and_return_conditional_losses_11559�
flatten/PartitionedCallPartitionedCalllReLU6/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *K
fFRD
B__inference_flatten_layer_call_and_return_conditional_losses_11567�
logVar/StatefulPartitionedCallStatefulPartitionedCall flatten/PartitionedCall:output:0logvar_11999logvar_12001*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������
*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *J
fERC
A__inference_logVar_layer_call_and_return_conditional_losses_11579�
mean/StatefulPartitionedCallStatefulPartitionedCall flatten/PartitionedCall:output:0
mean_12004
mean_12006*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������
*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *H
fCRA
?__inference_mean_layer_call_and_return_conditional_losses_11595t
IdentityIdentity%mean/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������
x

Identity_1Identity'logVar/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������
�
NoOpNoOp^bn1/StatefulPartitionedCall^bn2/StatefulPartitionedCall^bn3/StatefulPartitionedCall^bn4/StatefulPartitionedCall^bn5/StatefulPartitionedCall^bn6/StatefulPartitionedCall ^conv2D1/StatefulPartitionedCall ^conv2D2/StatefulPartitionedCall ^conv2D3/StatefulPartitionedCall ^conv2D4/StatefulPartitionedCall ^conv2D5/StatefulPartitionedCall ^conv2D6/StatefulPartitionedCall^logVar/StatefulPartitionedCall^mean/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*~
_input_shapesm
k:���������: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2:
bn1/StatefulPartitionedCallbn1/StatefulPartitionedCall2:
bn2/StatefulPartitionedCallbn2/StatefulPartitionedCall2:
bn3/StatefulPartitionedCallbn3/StatefulPartitionedCall2:
bn4/StatefulPartitionedCallbn4/StatefulPartitionedCall2:
bn5/StatefulPartitionedCallbn5/StatefulPartitionedCall2:
bn6/StatefulPartitionedCallbn6/StatefulPartitionedCall2B
conv2D1/StatefulPartitionedCallconv2D1/StatefulPartitionedCall2B
conv2D2/StatefulPartitionedCallconv2D2/StatefulPartitionedCall2B
conv2D3/StatefulPartitionedCallconv2D3/StatefulPartitionedCall2B
conv2D4/StatefulPartitionedCallconv2D4/StatefulPartitionedCall2B
conv2D5/StatefulPartitionedCallconv2D5/StatefulPartitionedCall2B
conv2D6/StatefulPartitionedCallconv2D6/StatefulPartitionedCall2@
logVar/StatefulPartitionedCalllogVar/StatefulPartitionedCall2<
mean/StatefulPartitionedCallmean/StatefulPartitionedCall:W S
/
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
>__inference_bn6_layer_call_and_return_conditional_losses_13481

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
�
B
&__inference_lReLU5_layer_call_fn_13395

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
A__inference_lReLU5_layer_call_and_return_conditional_losses_11527h
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
�
�
B__inference_encoder_layer_call_and_return_conditional_losses_12800

inputs@
&conv2d1_conv2d_readvariableop_resource: 5
'conv2d1_biasadd_readvariableop_resource: )
bn1_readvariableop_resource: +
bn1_readvariableop_1_resource: :
,bn1_fusedbatchnormv3_readvariableop_resource: <
.bn1_fusedbatchnormv3_readvariableop_1_resource: @
&conv2d2_conv2d_readvariableop_resource: @5
'conv2d2_biasadd_readvariableop_resource:@)
bn2_readvariableop_resource:@+
bn2_readvariableop_1_resource:@:
,bn2_fusedbatchnormv3_readvariableop_resource:@<
.bn2_fusedbatchnormv3_readvariableop_1_resource:@@
&conv2d3_conv2d_readvariableop_resource:@@5
'conv2d3_biasadd_readvariableop_resource:@)
bn3_readvariableop_resource:@+
bn3_readvariableop_1_resource:@:
,bn3_fusedbatchnormv3_readvariableop_resource:@<
.bn3_fusedbatchnormv3_readvariableop_1_resource:@@
&conv2d4_conv2d_readvariableop_resource:@@5
'conv2d4_biasadd_readvariableop_resource:@)
bn4_readvariableop_resource:@+
bn4_readvariableop_1_resource:@:
,bn4_fusedbatchnormv3_readvariableop_resource:@<
.bn4_fusedbatchnormv3_readvariableop_1_resource:@@
&conv2d5_conv2d_readvariableop_resource:@@5
'conv2d5_biasadd_readvariableop_resource:@)
bn5_readvariableop_resource:@+
bn5_readvariableop_1_resource:@:
,bn5_fusedbatchnormv3_readvariableop_resource:@<
.bn5_fusedbatchnormv3_readvariableop_1_resource:@@
&conv2d6_conv2d_readvariableop_resource:@ 5
'conv2d6_biasadd_readvariableop_resource: )
bn6_readvariableop_resource: +
bn6_readvariableop_1_resource: :
,bn6_fusedbatchnormv3_readvariableop_resource: <
.bn6_fusedbatchnormv3_readvariableop_1_resource: 8
%logvar_matmul_readvariableop_resource:	�
4
&logvar_biasadd_readvariableop_resource:
6
#mean_matmul_readvariableop_resource:	�
2
$mean_biasadd_readvariableop_resource:

identity

identity_1��bn1/AssignNewValue�bn1/AssignNewValue_1�#bn1/FusedBatchNormV3/ReadVariableOp�%bn1/FusedBatchNormV3/ReadVariableOp_1�bn1/ReadVariableOp�bn1/ReadVariableOp_1�bn2/AssignNewValue�bn2/AssignNewValue_1�#bn2/FusedBatchNormV3/ReadVariableOp�%bn2/FusedBatchNormV3/ReadVariableOp_1�bn2/ReadVariableOp�bn2/ReadVariableOp_1�bn3/AssignNewValue�bn3/AssignNewValue_1�#bn3/FusedBatchNormV3/ReadVariableOp�%bn3/FusedBatchNormV3/ReadVariableOp_1�bn3/ReadVariableOp�bn3/ReadVariableOp_1�bn4/AssignNewValue�bn4/AssignNewValue_1�#bn4/FusedBatchNormV3/ReadVariableOp�%bn4/FusedBatchNormV3/ReadVariableOp_1�bn4/ReadVariableOp�bn4/ReadVariableOp_1�bn5/AssignNewValue�bn5/AssignNewValue_1�#bn5/FusedBatchNormV3/ReadVariableOp�%bn5/FusedBatchNormV3/ReadVariableOp_1�bn5/ReadVariableOp�bn5/ReadVariableOp_1�bn6/AssignNewValue�bn6/AssignNewValue_1�#bn6/FusedBatchNormV3/ReadVariableOp�%bn6/FusedBatchNormV3/ReadVariableOp_1�bn6/ReadVariableOp�bn6/ReadVariableOp_1�conv2D1/BiasAdd/ReadVariableOp�conv2D1/Conv2D/ReadVariableOp�conv2D2/BiasAdd/ReadVariableOp�conv2D2/Conv2D/ReadVariableOp�conv2D3/BiasAdd/ReadVariableOp�conv2D3/Conv2D/ReadVariableOp�conv2D4/BiasAdd/ReadVariableOp�conv2D4/Conv2D/ReadVariableOp�conv2D5/BiasAdd/ReadVariableOp�conv2D5/Conv2D/ReadVariableOp�conv2D6/BiasAdd/ReadVariableOp�conv2D6/Conv2D/ReadVariableOp�logVar/BiasAdd/ReadVariableOp�logVar/MatMul/ReadVariableOp�mean/BiasAdd/ReadVariableOp�mean/MatMul/ReadVariableOp�
conv2D1/Conv2D/ReadVariableOpReadVariableOp&conv2d1_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype0�
conv2D1/Conv2DConv2Dinputs%conv2D1/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:��������� *
paddingSAME*
strides
�
conv2D1/BiasAdd/ReadVariableOpReadVariableOp'conv2d1_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0�
conv2D1/BiasAddBiasAddconv2D1/Conv2D:output:0&conv2D1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:��������� j
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
bn1/FusedBatchNormV3FusedBatchNormV3conv2D1/BiasAdd:output:0bn1/ReadVariableOp:value:0bn1/ReadVariableOp_1:value:0+bn1/FusedBatchNormV3/ReadVariableOp:value:0-bn1/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:��������� : : : : :*
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
:��������� *
alpha%���>�
conv2D2/Conv2D/ReadVariableOpReadVariableOp&conv2d2_conv2d_readvariableop_resource*&
_output_shapes
: @*
dtype0�
conv2D2/Conv2DConv2DlReLU1/LeakyRelu:activations:0%conv2D2/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@*
paddingSAME*
strides
�
conv2D2/BiasAdd/ReadVariableOpReadVariableOp'conv2d2_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0�
conv2D2/BiasAddBiasAddconv2D2/Conv2D:output:0&conv2D2/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@j
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
bn2/FusedBatchNormV3FusedBatchNormV3conv2D2/BiasAdd:output:0bn2/ReadVariableOp:value:0bn2/ReadVariableOp_1:value:0+bn2/FusedBatchNormV3/ReadVariableOp:value:0-bn2/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:���������@:@:@:@:@:*
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
:���������@*
alpha%���>�
conv2D3/Conv2D/ReadVariableOpReadVariableOp&conv2d3_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype0�
conv2D3/Conv2DConv2DlReLU2/LeakyRelu:activations:0%conv2D3/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@*
paddingSAME*
strides
�
conv2D3/BiasAdd/ReadVariableOpReadVariableOp'conv2d3_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0�
conv2D3/BiasAddBiasAddconv2D3/Conv2D:output:0&conv2D3/BiasAdd/ReadVariableOp:value:0*
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
bn3/FusedBatchNormV3FusedBatchNormV3conv2D3/BiasAdd:output:0bn3/ReadVariableOp:value:0bn3/ReadVariableOp_1:value:0+bn3/FusedBatchNormV3/ReadVariableOp:value:0-bn3/FusedBatchNormV3/ReadVariableOp_1:value:0*
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
alpha%���>�
conv2D4/Conv2D/ReadVariableOpReadVariableOp&conv2d4_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype0�
conv2D4/Conv2DConv2DlReLU3/LeakyRelu:activations:0%conv2D4/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@*
paddingSAME*
strides
�
conv2D4/BiasAdd/ReadVariableOpReadVariableOp'conv2d4_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0�
conv2D4/BiasAddBiasAddconv2D4/Conv2D:output:0&conv2D4/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@j
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
bn4/FusedBatchNormV3FusedBatchNormV3conv2D4/BiasAdd:output:0bn4/ReadVariableOp:value:0bn4/ReadVariableOp_1:value:0+bn4/FusedBatchNormV3/ReadVariableOp:value:0-bn4/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:���������@:@:@:@:@:*
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
:���������@*
alpha%���>�
conv2D5/Conv2D/ReadVariableOpReadVariableOp&conv2d5_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype0�
conv2D5/Conv2DConv2DlReLU4/LeakyRelu:activations:0%conv2D5/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@*
paddingSAME*
strides
�
conv2D5/BiasAdd/ReadVariableOpReadVariableOp'conv2d5_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0�
conv2D5/BiasAddBiasAddconv2D5/Conv2D:output:0&conv2D5/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@j
bn5/ReadVariableOpReadVariableOpbn5_readvariableop_resource*
_output_shapes
:@*
dtype0n
bn5/ReadVariableOp_1ReadVariableOpbn5_readvariableop_1_resource*
_output_shapes
:@*
dtype0�
#bn5/FusedBatchNormV3/ReadVariableOpReadVariableOp,bn5_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype0�
%bn5/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp.bn5_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype0�
bn5/FusedBatchNormV3FusedBatchNormV3conv2D5/BiasAdd:output:0bn5/ReadVariableOp:value:0bn5/ReadVariableOp_1:value:0+bn5/FusedBatchNormV3/ReadVariableOp:value:0-bn5/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:���������@:@:@:@:@:*
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
:���������@*
alpha%���>�
conv2D6/Conv2D/ReadVariableOpReadVariableOp&conv2d6_conv2d_readvariableop_resource*&
_output_shapes
:@ *
dtype0�
conv2D6/Conv2DConv2DlReLU5/LeakyRelu:activations:0%conv2D6/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:��������� *
paddingSAME*
strides
�
conv2D6/BiasAdd/ReadVariableOpReadVariableOp'conv2d6_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0�
conv2D6/BiasAddBiasAddconv2D6/Conv2D:output:0&conv2D6/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:��������� j
bn6/ReadVariableOpReadVariableOpbn6_readvariableop_resource*
_output_shapes
: *
dtype0n
bn6/ReadVariableOp_1ReadVariableOpbn6_readvariableop_1_resource*
_output_shapes
: *
dtype0�
#bn6/FusedBatchNormV3/ReadVariableOpReadVariableOp,bn6_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype0�
%bn6/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp.bn6_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype0�
bn6/FusedBatchNormV3FusedBatchNormV3conv2D6/BiasAdd:output:0bn6/ReadVariableOp:value:0bn6/ReadVariableOp_1:value:0+bn6/FusedBatchNormV3/ReadVariableOp:value:0-bn6/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:��������� : : : : :*
epsilon%o�:*
exponential_avg_factor%
�#<�
bn6/AssignNewValueAssignVariableOp,bn6_fusedbatchnormv3_readvariableop_resource!bn6/FusedBatchNormV3:batch_mean:0$^bn6/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(�
bn6/AssignNewValue_1AssignVariableOp.bn6_fusedbatchnormv3_readvariableop_1_resource%bn6/FusedBatchNormV3:batch_variance:0&^bn6/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(x
lReLU6/LeakyRelu	LeakyRelubn6/FusedBatchNormV3:y:0*/
_output_shapes
:��������� *
alpha%���>^
flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"����   �
flatten/ReshapeReshapelReLU6/LeakyRelu:activations:0flatten/Const:output:0*
T0*(
_output_shapes
:�����������
logVar/MatMul/ReadVariableOpReadVariableOp%logvar_matmul_readvariableop_resource*
_output_shapes
:	�
*
dtype0�
logVar/MatMulMatMulflatten/Reshape:output:0$logVar/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������
�
logVar/BiasAdd/ReadVariableOpReadVariableOp&logvar_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype0�
logVar/BiasAddBiasAddlogVar/MatMul:product:0%logVar/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������

mean/MatMul/ReadVariableOpReadVariableOp#mean_matmul_readvariableop_resource*
_output_shapes
:	�
*
dtype0�
mean/MatMulMatMulflatten/Reshape:output:0"mean/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������
|
mean/BiasAdd/ReadVariableOpReadVariableOp$mean_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype0�
mean/BiasAddBiasAddmean/MatMul:product:0#mean/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������
d
IdentityIdentitymean/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:���������
h

Identity_1IdentitylogVar/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:���������
�
NoOpNoOp^bn1/AssignNewValue^bn1/AssignNewValue_1$^bn1/FusedBatchNormV3/ReadVariableOp&^bn1/FusedBatchNormV3/ReadVariableOp_1^bn1/ReadVariableOp^bn1/ReadVariableOp_1^bn2/AssignNewValue^bn2/AssignNewValue_1$^bn2/FusedBatchNormV3/ReadVariableOp&^bn2/FusedBatchNormV3/ReadVariableOp_1^bn2/ReadVariableOp^bn2/ReadVariableOp_1^bn3/AssignNewValue^bn3/AssignNewValue_1$^bn3/FusedBatchNormV3/ReadVariableOp&^bn3/FusedBatchNormV3/ReadVariableOp_1^bn3/ReadVariableOp^bn3/ReadVariableOp_1^bn4/AssignNewValue^bn4/AssignNewValue_1$^bn4/FusedBatchNormV3/ReadVariableOp&^bn4/FusedBatchNormV3/ReadVariableOp_1^bn4/ReadVariableOp^bn4/ReadVariableOp_1^bn5/AssignNewValue^bn5/AssignNewValue_1$^bn5/FusedBatchNormV3/ReadVariableOp&^bn5/FusedBatchNormV3/ReadVariableOp_1^bn5/ReadVariableOp^bn5/ReadVariableOp_1^bn6/AssignNewValue^bn6/AssignNewValue_1$^bn6/FusedBatchNormV3/ReadVariableOp&^bn6/FusedBatchNormV3/ReadVariableOp_1^bn6/ReadVariableOp^bn6/ReadVariableOp_1^conv2D1/BiasAdd/ReadVariableOp^conv2D1/Conv2D/ReadVariableOp^conv2D2/BiasAdd/ReadVariableOp^conv2D2/Conv2D/ReadVariableOp^conv2D3/BiasAdd/ReadVariableOp^conv2D3/Conv2D/ReadVariableOp^conv2D4/BiasAdd/ReadVariableOp^conv2D4/Conv2D/ReadVariableOp^conv2D5/BiasAdd/ReadVariableOp^conv2D5/Conv2D/ReadVariableOp^conv2D6/BiasAdd/ReadVariableOp^conv2D6/Conv2D/ReadVariableOp^logVar/BiasAdd/ReadVariableOp^logVar/MatMul/ReadVariableOp^mean/BiasAdd/ReadVariableOp^mean/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*~
_input_shapesm
k:���������: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2(
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
bn5/ReadVariableOp_1bn5/ReadVariableOp_12(
bn6/AssignNewValuebn6/AssignNewValue2,
bn6/AssignNewValue_1bn6/AssignNewValue_12J
#bn6/FusedBatchNormV3/ReadVariableOp#bn6/FusedBatchNormV3/ReadVariableOp2N
%bn6/FusedBatchNormV3/ReadVariableOp_1%bn6/FusedBatchNormV3/ReadVariableOp_12(
bn6/ReadVariableOpbn6/ReadVariableOp2,
bn6/ReadVariableOp_1bn6/ReadVariableOp_12@
conv2D1/BiasAdd/ReadVariableOpconv2D1/BiasAdd/ReadVariableOp2>
conv2D1/Conv2D/ReadVariableOpconv2D1/Conv2D/ReadVariableOp2@
conv2D2/BiasAdd/ReadVariableOpconv2D2/BiasAdd/ReadVariableOp2>
conv2D2/Conv2D/ReadVariableOpconv2D2/Conv2D/ReadVariableOp2@
conv2D3/BiasAdd/ReadVariableOpconv2D3/BiasAdd/ReadVariableOp2>
conv2D3/Conv2D/ReadVariableOpconv2D3/Conv2D/ReadVariableOp2@
conv2D4/BiasAdd/ReadVariableOpconv2D4/BiasAdd/ReadVariableOp2>
conv2D4/Conv2D/ReadVariableOpconv2D4/Conv2D/ReadVariableOp2@
conv2D5/BiasAdd/ReadVariableOpconv2D5/BiasAdd/ReadVariableOp2>
conv2D5/Conv2D/ReadVariableOpconv2D5/Conv2D/ReadVariableOp2@
conv2D6/BiasAdd/ReadVariableOpconv2D6/BiasAdd/ReadVariableOp2>
conv2D6/Conv2D/ReadVariableOpconv2D6/Conv2D/ReadVariableOp2>
logVar/BiasAdd/ReadVariableOplogVar/BiasAdd/ReadVariableOp2<
logVar/MatMul/ReadVariableOplogVar/MatMul/ReadVariableOp2:
mean/BiasAdd/ReadVariableOpmean/BiasAdd/ReadVariableOp28
mean/MatMul/ReadVariableOpmean/MatMul/ReadVariableOp:W S
/
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
'__inference_conv2D4_layer_call_fn_13227

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
 */
_output_shapes
:���������@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *K
fFRD
B__inference_conv2D4_layer_call_and_return_conditional_losses_11475w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:���������@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������@: : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:���������@
 
_user_specified_nameinputs
�
B
&__inference_lReLU3_layer_call_fn_13213

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
A__inference_lReLU3_layer_call_and_return_conditional_losses_11463h
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
K
input_layer<
serving_default_input_layer:0���������:
logVar0
StatefulPartitionedCall:0���������
8
mean0
StatefulPartitionedCall:1���������
tensorflow/serving/predict:��
�
layer-0
layer_with_weights-0
layer-1
layer_with_weights-1
layer-2
layer-3
layer_with_weights-2
layer-4
layer_with_weights-3
layer-5
layer-6
layer_with_weights-4
layer-7
	layer_with_weights-5
	layer-8

layer-9
layer_with_weights-6
layer-10
layer_with_weights-7
layer-11
layer-12
layer_with_weights-8
layer-13
layer_with_weights-9
layer-14
layer-15
layer_with_weights-10
layer-16
layer_with_weights-11
layer-17
layer-18
layer-19
layer_with_weights-12
layer-20
layer_with_weights-13
layer-21
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
_default_save_signature

signatures"
_tf_keras_network
"
_tf_keras_input_layer
�
	variables
 trainable_variables
!regularization_losses
"	keras_api
#__call__
*$&call_and_return_all_conditional_losses

%kernel
&bias
 '_jit_compiled_convolution_op"
_tf_keras_layer
�
(	variables
)trainable_variables
*regularization_losses
+	keras_api
,__call__
*-&call_and_return_all_conditional_losses
.axis
	/gamma
0beta
1moving_mean
2moving_variance"
_tf_keras_layer
�
3	variables
4trainable_variables
5regularization_losses
6	keras_api
7__call__
*8&call_and_return_all_conditional_losses"
_tf_keras_layer
�
9	variables
:trainable_variables
;regularization_losses
<	keras_api
=__call__
*>&call_and_return_all_conditional_losses

?kernel
@bias
 A_jit_compiled_convolution_op"
_tf_keras_layer
�
B	variables
Ctrainable_variables
Dregularization_losses
E	keras_api
F__call__
*G&call_and_return_all_conditional_losses
Haxis
	Igamma
Jbeta
Kmoving_mean
Lmoving_variance"
_tf_keras_layer
�
M	variables
Ntrainable_variables
Oregularization_losses
P	keras_api
Q__call__
*R&call_and_return_all_conditional_losses"
_tf_keras_layer
�
S	variables
Ttrainable_variables
Uregularization_losses
V	keras_api
W__call__
*X&call_and_return_all_conditional_losses

Ykernel
Zbias
 [_jit_compiled_convolution_op"
_tf_keras_layer
�
\	variables
]trainable_variables
^regularization_losses
_	keras_api
`__call__
*a&call_and_return_all_conditional_losses
baxis
	cgamma
dbeta
emoving_mean
fmoving_variance"
_tf_keras_layer
�
g	variables
htrainable_variables
iregularization_losses
j	keras_api
k__call__
*l&call_and_return_all_conditional_losses"
_tf_keras_layer
�
m	variables
ntrainable_variables
oregularization_losses
p	keras_api
q__call__
*r&call_and_return_all_conditional_losses

skernel
tbias
 u_jit_compiled_convolution_op"
_tf_keras_layer
�
v	variables
wtrainable_variables
xregularization_losses
y	keras_api
z__call__
*{&call_and_return_all_conditional_losses
|axis
	}gamma
~beta
moving_mean
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
	�bias"
_tf_keras_layer
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�kernel
	�bias"
_tf_keras_layer
�
%0
&1
/2
03
14
25
?6
@7
I8
J9
K10
L11
Y12
Z13
c14
d15
e16
f17
s18
t19
}20
~21
22
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
�33
�34
�35
�36
�37
�38
�39"
trackable_list_wrapper
�
%0
&1
/2
03
?4
@5
I6
J7
Y8
Z9
c10
d11
s12
t13
}14
~15
�16
�17
�18
�19
�20
�21
�22
�23
�24
�25
�26
�27"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
	variables
trainable_variables
regularization_losses
__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
�
�trace_0
�trace_1
�trace_2
�trace_32�
'__inference_encoder_layer_call_fn_11903
'__inference_encoder_layer_call_fn_12096
'__inference_encoder_layer_call_fn_12568
'__inference_encoder_layer_call_fn_12655�
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
B__inference_encoder_layer_call_and_return_conditional_losses_11603
B__inference_encoder_layer_call_and_return_conditional_losses_11709
B__inference_encoder_layer_call_and_return_conditional_losses_12800
B__inference_encoder_layer_call_and_return_conditional_losses_12945�
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
 __inference__wrapped_model_10981input_layer"�
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
%0
&1"
trackable_list_wrapper
.
%0
&1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
	variables
 trainable_variables
!regularization_losses
#__call__
*$&call_and_return_all_conditional_losses
&$"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
'__inference_conv2D1_layer_call_fn_12954�
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
B__inference_conv2D1_layer_call_and_return_conditional_losses_12964�
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
(:& 2conv2D1/kernel
: 2conv2D1/bias
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
/0
01
12
23"
trackable_list_wrapper
.
/0
01"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
(	variables
)trainable_variables
*regularization_losses
,__call__
*-&call_and_return_all_conditional_losses
&-"call_and_return_conditional_losses"
_generic_user_object
�
�trace_0
�trace_12�
#__inference_bn1_layer_call_fn_12977
#__inference_bn1_layer_call_fn_12990�
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
>__inference_bn1_layer_call_and_return_conditional_losses_13008
>__inference_bn1_layer_call_and_return_conditional_losses_13026�
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
3	variables
4trainable_variables
5regularization_losses
7__call__
*8&call_and_return_all_conditional_losses
&8"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
&__inference_lReLU1_layer_call_fn_13031�
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
A__inference_lReLU1_layer_call_and_return_conditional_losses_13036�
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
?0
@1"
trackable_list_wrapper
.
?0
@1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
9	variables
:trainable_variables
;regularization_losses
=__call__
*>&call_and_return_all_conditional_losses
&>"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
'__inference_conv2D2_layer_call_fn_13045�
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
B__inference_conv2D2_layer_call_and_return_conditional_losses_13055�
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
(:& @2conv2D2/kernel
:@2conv2D2/bias
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
I0
J1
K2
L3"
trackable_list_wrapper
.
I0
J1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
B	variables
Ctrainable_variables
Dregularization_losses
F__call__
*G&call_and_return_all_conditional_losses
&G"call_and_return_conditional_losses"
_generic_user_object
�
�trace_0
�trace_12�
#__inference_bn2_layer_call_fn_13068
#__inference_bn2_layer_call_fn_13081�
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
>__inference_bn2_layer_call_and_return_conditional_losses_13099
>__inference_bn2_layer_call_and_return_conditional_losses_13117�
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
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
M	variables
Ntrainable_variables
Oregularization_losses
Q__call__
*R&call_and_return_all_conditional_losses
&R"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
&__inference_lReLU2_layer_call_fn_13122�
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
A__inference_lReLU2_layer_call_and_return_conditional_losses_13127�
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
Y0
Z1"
trackable_list_wrapper
.
Y0
Z1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
S	variables
Ttrainable_variables
Uregularization_losses
W__call__
*X&call_and_return_all_conditional_losses
&X"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
'__inference_conv2D3_layer_call_fn_13136�
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
B__inference_conv2D3_layer_call_and_return_conditional_losses_13146�
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
(:&@@2conv2D3/kernel
:@2conv2D3/bias
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
c0
d1
e2
f3"
trackable_list_wrapper
.
c0
d1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
\	variables
]trainable_variables
^regularization_losses
`__call__
*a&call_and_return_all_conditional_losses
&a"call_and_return_conditional_losses"
_generic_user_object
�
�trace_0
�trace_12�
#__inference_bn3_layer_call_fn_13159
#__inference_bn3_layer_call_fn_13172�
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
>__inference_bn3_layer_call_and_return_conditional_losses_13190
>__inference_bn3_layer_call_and_return_conditional_losses_13208�
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
g	variables
htrainable_variables
iregularization_losses
k__call__
*l&call_and_return_all_conditional_losses
&l"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
&__inference_lReLU3_layer_call_fn_13213�
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
A__inference_lReLU3_layer_call_and_return_conditional_losses_13218�
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
s0
t1"
trackable_list_wrapper
.
s0
t1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
m	variables
ntrainable_variables
oregularization_losses
q__call__
*r&call_and_return_all_conditional_losses
&r"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
'__inference_conv2D4_layer_call_fn_13227�
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
B__inference_conv2D4_layer_call_and_return_conditional_losses_13237�
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
(:&@@2conv2D4/kernel
:@2conv2D4/bias
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
=
}0
~1
2
�3"
trackable_list_wrapper
.
}0
~1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
v	variables
wtrainable_variables
xregularization_losses
z__call__
*{&call_and_return_all_conditional_losses
&{"call_and_return_conditional_losses"
_generic_user_object
�
�trace_0
�trace_12�
#__inference_bn4_layer_call_fn_13250
#__inference_bn4_layer_call_fn_13263�
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
>__inference_bn4_layer_call_and_return_conditional_losses_13281
>__inference_bn4_layer_call_and_return_conditional_losses_13299�
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
&__inference_lReLU4_layer_call_fn_13304�
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
A__inference_lReLU4_layer_call_and_return_conditional_losses_13309�
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
'__inference_conv2D5_layer_call_fn_13318�
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
B__inference_conv2D5_layer_call_and_return_conditional_losses_13328�
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
(:&@@2conv2D5/kernel
:@2conv2D5/bias
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
#__inference_bn5_layer_call_fn_13341
#__inference_bn5_layer_call_fn_13354�
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
>__inference_bn5_layer_call_and_return_conditional_losses_13372
>__inference_bn5_layer_call_and_return_conditional_losses_13390�
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
:@2	bn5/gamma
:@2bn5/beta
:@ (2bn5/moving_mean
#:!@ (2bn5/moving_variance
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
&__inference_lReLU5_layer_call_fn_13395�
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
A__inference_lReLU5_layer_call_and_return_conditional_losses_13400�
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
'__inference_conv2D6_layer_call_fn_13409�
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
B__inference_conv2D6_layer_call_and_return_conditional_losses_13419�
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
(:&@ 2conv2D6/kernel
: 2conv2D6/bias
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
#__inference_bn6_layer_call_fn_13432
#__inference_bn6_layer_call_fn_13445�
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
>__inference_bn6_layer_call_and_return_conditional_losses_13463
>__inference_bn6_layer_call_and_return_conditional_losses_13481�
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
: 2	bn6/gamma
: 2bn6/beta
:  (2bn6/moving_mean
#:!  (2bn6/moving_variance
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
&__inference_lReLU6_layer_call_fn_13486�
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
A__inference_lReLU6_layer_call_and_return_conditional_losses_13491�
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
'__inference_flatten_layer_call_fn_13496�
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
B__inference_flatten_layer_call_and_return_conditional_losses_13502�
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
$__inference_mean_layer_call_fn_13511�
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
?__inference_mean_layer_call_and_return_conditional_losses_13521�
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
:	�
2mean/kernel
:
2	mean/bias
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
&__inference_logVar_layer_call_fn_13530�
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
A__inference_logVar_layer_call_and_return_conditional_losses_13540�
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
 :	�
2logVar/kernel
:
2logVar/bias
{
10
21
K2
L3
e4
f5
6
�7
�8
�9
�10
�11"
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
18
19
20
21"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
'__inference_encoder_layer_call_fn_11903input_layer"�
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
'__inference_encoder_layer_call_fn_12096input_layer"�
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
'__inference_encoder_layer_call_fn_12568inputs"�
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
'__inference_encoder_layer_call_fn_12655inputs"�
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
B__inference_encoder_layer_call_and_return_conditional_losses_11603input_layer"�
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
B__inference_encoder_layer_call_and_return_conditional_losses_11709input_layer"�
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
B__inference_encoder_layer_call_and_return_conditional_losses_12800inputs"�
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
B__inference_encoder_layer_call_and_return_conditional_losses_12945inputs"�
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
#__inference_signature_wrapper_12481input_layer"�
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
'__inference_conv2D1_layer_call_fn_12954inputs"�
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
B__inference_conv2D1_layer_call_and_return_conditional_losses_12964inputs"�
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
10
21"
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
#__inference_bn1_layer_call_fn_12977inputs"�
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
#__inference_bn1_layer_call_fn_12990inputs"�
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
>__inference_bn1_layer_call_and_return_conditional_losses_13008inputs"�
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
>__inference_bn1_layer_call_and_return_conditional_losses_13026inputs"�
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
&__inference_lReLU1_layer_call_fn_13031inputs"�
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
A__inference_lReLU1_layer_call_and_return_conditional_losses_13036inputs"�
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
'__inference_conv2D2_layer_call_fn_13045inputs"�
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
B__inference_conv2D2_layer_call_and_return_conditional_losses_13055inputs"�
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
K0
L1"
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
#__inference_bn2_layer_call_fn_13068inputs"�
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
#__inference_bn2_layer_call_fn_13081inputs"�
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
>__inference_bn2_layer_call_and_return_conditional_losses_13099inputs"�
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
>__inference_bn2_layer_call_and_return_conditional_losses_13117inputs"�
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
&__inference_lReLU2_layer_call_fn_13122inputs"�
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
A__inference_lReLU2_layer_call_and_return_conditional_losses_13127inputs"�
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
'__inference_conv2D3_layer_call_fn_13136inputs"�
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
B__inference_conv2D3_layer_call_and_return_conditional_losses_13146inputs"�
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
e0
f1"
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
#__inference_bn3_layer_call_fn_13159inputs"�
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
#__inference_bn3_layer_call_fn_13172inputs"�
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
>__inference_bn3_layer_call_and_return_conditional_losses_13190inputs"�
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
>__inference_bn3_layer_call_and_return_conditional_losses_13208inputs"�
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
&__inference_lReLU3_layer_call_fn_13213inputs"�
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
A__inference_lReLU3_layer_call_and_return_conditional_losses_13218inputs"�
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
'__inference_conv2D4_layer_call_fn_13227inputs"�
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
B__inference_conv2D4_layer_call_and_return_conditional_losses_13237inputs"�
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
/
0
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
#__inference_bn4_layer_call_fn_13250inputs"�
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
#__inference_bn4_layer_call_fn_13263inputs"�
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
>__inference_bn4_layer_call_and_return_conditional_losses_13281inputs"�
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
>__inference_bn4_layer_call_and_return_conditional_losses_13299inputs"�
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
&__inference_lReLU4_layer_call_fn_13304inputs"�
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
A__inference_lReLU4_layer_call_and_return_conditional_losses_13309inputs"�
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
'__inference_conv2D5_layer_call_fn_13318inputs"�
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
B__inference_conv2D5_layer_call_and_return_conditional_losses_13328inputs"�
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
#__inference_bn5_layer_call_fn_13341inputs"�
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
#__inference_bn5_layer_call_fn_13354inputs"�
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
>__inference_bn5_layer_call_and_return_conditional_losses_13372inputs"�
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
>__inference_bn5_layer_call_and_return_conditional_losses_13390inputs"�
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
&__inference_lReLU5_layer_call_fn_13395inputs"�
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
A__inference_lReLU5_layer_call_and_return_conditional_losses_13400inputs"�
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
'__inference_conv2D6_layer_call_fn_13409inputs"�
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
B__inference_conv2D6_layer_call_and_return_conditional_losses_13419inputs"�
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
#__inference_bn6_layer_call_fn_13432inputs"�
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
#__inference_bn6_layer_call_fn_13445inputs"�
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
>__inference_bn6_layer_call_and_return_conditional_losses_13463inputs"�
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
>__inference_bn6_layer_call_and_return_conditional_losses_13481inputs"�
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
&__inference_lReLU6_layer_call_fn_13486inputs"�
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
A__inference_lReLU6_layer_call_and_return_conditional_losses_13491inputs"�
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
'__inference_flatten_layer_call_fn_13496inputs"�
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
B__inference_flatten_layer_call_and_return_conditional_losses_13502inputs"�
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
$__inference_mean_layer_call_fn_13511inputs"�
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
?__inference_mean_layer_call_and_return_conditional_losses_13521inputs"�
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
&__inference_logVar_layer_call_fn_13530inputs"�
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
A__inference_logVar_layer_call_and_return_conditional_losses_13540inputs"�
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
 __inference__wrapped_model_10981�9%&/012?@IJKLYZcdefst}~�����������������<�9
2�/
-�*
input_layer���������
� "W�T
*
logVar �
logvar���������

&
mean�
mean���������
�
>__inference_bn1_layer_call_and_return_conditional_losses_13008�/012Q�N
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
>__inference_bn1_layer_call_and_return_conditional_losses_13026�/012Q�N
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
#__inference_bn1_layer_call_fn_12977�/012Q�N
G�D
:�7
inputs+��������������������������� 
p

 
� ";�8
unknown+��������������������������� �
#__inference_bn1_layer_call_fn_12990�/012Q�N
G�D
:�7
inputs+��������������������������� 
p 

 
� ";�8
unknown+��������������������������� �
>__inference_bn2_layer_call_and_return_conditional_losses_13099�IJKLQ�N
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
>__inference_bn2_layer_call_and_return_conditional_losses_13117�IJKLQ�N
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
#__inference_bn2_layer_call_fn_13068�IJKLQ�N
G�D
:�7
inputs+���������������������������@
p

 
� ";�8
unknown+���������������������������@�
#__inference_bn2_layer_call_fn_13081�IJKLQ�N
G�D
:�7
inputs+���������������������������@
p 

 
� ";�8
unknown+���������������������������@�
>__inference_bn3_layer_call_and_return_conditional_losses_13190�cdefQ�N
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
>__inference_bn3_layer_call_and_return_conditional_losses_13208�cdefQ�N
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
#__inference_bn3_layer_call_fn_13159�cdefQ�N
G�D
:�7
inputs+���������������������������@
p

 
� ";�8
unknown+���������������������������@�
#__inference_bn3_layer_call_fn_13172�cdefQ�N
G�D
:�7
inputs+���������������������������@
p 

 
� ";�8
unknown+���������������������������@�
>__inference_bn4_layer_call_and_return_conditional_losses_13281�}~�Q�N
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
>__inference_bn4_layer_call_and_return_conditional_losses_13299�}~�Q�N
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
#__inference_bn4_layer_call_fn_13250�}~�Q�N
G�D
:�7
inputs+���������������������������@
p

 
� ";�8
unknown+���������������������������@�
#__inference_bn4_layer_call_fn_13263�}~�Q�N
G�D
:�7
inputs+���������������������������@
p 

 
� ";�8
unknown+���������������������������@�
>__inference_bn5_layer_call_and_return_conditional_losses_13372�����Q�N
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
>__inference_bn5_layer_call_and_return_conditional_losses_13390�����Q�N
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
#__inference_bn5_layer_call_fn_13341�����Q�N
G�D
:�7
inputs+���������������������������@
p

 
� ";�8
unknown+���������������������������@�
#__inference_bn5_layer_call_fn_13354�����Q�N
G�D
:�7
inputs+���������������������������@
p 

 
� ";�8
unknown+���������������������������@�
>__inference_bn6_layer_call_and_return_conditional_losses_13463�����Q�N
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
>__inference_bn6_layer_call_and_return_conditional_losses_13481�����Q�N
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
#__inference_bn6_layer_call_fn_13432�����Q�N
G�D
:�7
inputs+��������������������������� 
p

 
� ";�8
unknown+��������������������������� �
#__inference_bn6_layer_call_fn_13445�����Q�N
G�D
:�7
inputs+��������������������������� 
p 

 
� ";�8
unknown+��������������������������� �
B__inference_conv2D1_layer_call_and_return_conditional_losses_12964s%&7�4
-�*
(�%
inputs���������
� "4�1
*�'
tensor_0��������� 
� �
'__inference_conv2D1_layer_call_fn_12954h%&7�4
-�*
(�%
inputs���������
� ")�&
unknown��������� �
B__inference_conv2D2_layer_call_and_return_conditional_losses_13055s?@7�4
-�*
(�%
inputs��������� 
� "4�1
*�'
tensor_0���������@
� �
'__inference_conv2D2_layer_call_fn_13045h?@7�4
-�*
(�%
inputs��������� 
� ")�&
unknown���������@�
B__inference_conv2D3_layer_call_and_return_conditional_losses_13146sYZ7�4
-�*
(�%
inputs���������@
� "4�1
*�'
tensor_0���������@
� �
'__inference_conv2D3_layer_call_fn_13136hYZ7�4
-�*
(�%
inputs���������@
� ")�&
unknown���������@�
B__inference_conv2D4_layer_call_and_return_conditional_losses_13237sst7�4
-�*
(�%
inputs���������@
� "4�1
*�'
tensor_0���������@
� �
'__inference_conv2D4_layer_call_fn_13227hst7�4
-�*
(�%
inputs���������@
� ")�&
unknown���������@�
B__inference_conv2D5_layer_call_and_return_conditional_losses_13328u��7�4
-�*
(�%
inputs���������@
� "4�1
*�'
tensor_0���������@
� �
'__inference_conv2D5_layer_call_fn_13318j��7�4
-�*
(�%
inputs���������@
� ")�&
unknown���������@�
B__inference_conv2D6_layer_call_and_return_conditional_losses_13419u��7�4
-�*
(�%
inputs���������@
� "4�1
*�'
tensor_0��������� 
� �
'__inference_conv2D6_layer_call_fn_13409j��7�4
-�*
(�%
inputs���������@
� ")�&
unknown��������� �
B__inference_encoder_layer_call_and_return_conditional_losses_11603�9%&/012?@IJKLYZcdefst}~�����������������D�A
:�7
-�*
input_layer���������
p

 
� "Y�V
O�L
$�!

tensor_0_0���������

$�!

tensor_0_1���������

� �
B__inference_encoder_layer_call_and_return_conditional_losses_11709�9%&/012?@IJKLYZcdefst}~�����������������D�A
:�7
-�*
input_layer���������
p 

 
� "Y�V
O�L
$�!

tensor_0_0���������

$�!

tensor_0_1���������

� �
B__inference_encoder_layer_call_and_return_conditional_losses_12800�9%&/012?@IJKLYZcdefst}~�����������������?�<
5�2
(�%
inputs���������
p

 
� "Y�V
O�L
$�!

tensor_0_0���������

$�!

tensor_0_1���������

� �
B__inference_encoder_layer_call_and_return_conditional_losses_12945�9%&/012?@IJKLYZcdefst}~�����������������?�<
5�2
(�%
inputs���������
p 

 
� "Y�V
O�L
$�!

tensor_0_0���������

$�!

tensor_0_1���������

� �
'__inference_encoder_layer_call_fn_11903�9%&/012?@IJKLYZcdefst}~�����������������D�A
:�7
-�*
input_layer���������
p

 
� "K�H
"�
tensor_0���������

"�
tensor_1���������
�
'__inference_encoder_layer_call_fn_12096�9%&/012?@IJKLYZcdefst}~�����������������D�A
:�7
-�*
input_layer���������
p 

 
� "K�H
"�
tensor_0���������

"�
tensor_1���������
�
'__inference_encoder_layer_call_fn_12568�9%&/012?@IJKLYZcdefst}~�����������������?�<
5�2
(�%
inputs���������
p

 
� "K�H
"�
tensor_0���������

"�
tensor_1���������
�
'__inference_encoder_layer_call_fn_12655�9%&/012?@IJKLYZcdefst}~�����������������?�<
5�2
(�%
inputs���������
p 

 
� "K�H
"�
tensor_0���������

"�
tensor_1���������
�
B__inference_flatten_layer_call_and_return_conditional_losses_13502h7�4
-�*
(�%
inputs��������� 
� "-�*
#� 
tensor_0����������
� �
'__inference_flatten_layer_call_fn_13496]7�4
-�*
(�%
inputs��������� 
� ""�
unknown�����������
A__inference_lReLU1_layer_call_and_return_conditional_losses_13036o7�4
-�*
(�%
inputs��������� 
� "4�1
*�'
tensor_0��������� 
� �
&__inference_lReLU1_layer_call_fn_13031d7�4
-�*
(�%
inputs��������� 
� ")�&
unknown��������� �
A__inference_lReLU2_layer_call_and_return_conditional_losses_13127o7�4
-�*
(�%
inputs���������@
� "4�1
*�'
tensor_0���������@
� �
&__inference_lReLU2_layer_call_fn_13122d7�4
-�*
(�%
inputs���������@
� ")�&
unknown���������@�
A__inference_lReLU3_layer_call_and_return_conditional_losses_13218o7�4
-�*
(�%
inputs���������@
� "4�1
*�'
tensor_0���������@
� �
&__inference_lReLU3_layer_call_fn_13213d7�4
-�*
(�%
inputs���������@
� ")�&
unknown���������@�
A__inference_lReLU4_layer_call_and_return_conditional_losses_13309o7�4
-�*
(�%
inputs���������@
� "4�1
*�'
tensor_0���������@
� �
&__inference_lReLU4_layer_call_fn_13304d7�4
-�*
(�%
inputs���������@
� ")�&
unknown���������@�
A__inference_lReLU5_layer_call_and_return_conditional_losses_13400o7�4
-�*
(�%
inputs���������@
� "4�1
*�'
tensor_0���������@
� �
&__inference_lReLU5_layer_call_fn_13395d7�4
-�*
(�%
inputs���������@
� ")�&
unknown���������@�
A__inference_lReLU6_layer_call_and_return_conditional_losses_13491o7�4
-�*
(�%
inputs��������� 
� "4�1
*�'
tensor_0��������� 
� �
&__inference_lReLU6_layer_call_fn_13486d7�4
-�*
(�%
inputs��������� 
� ")�&
unknown��������� �
A__inference_logVar_layer_call_and_return_conditional_losses_13540f��0�-
&�#
!�
inputs����������
� ",�)
"�
tensor_0���������

� �
&__inference_logVar_layer_call_fn_13530[��0�-
&�#
!�
inputs����������
� "!�
unknown���������
�
?__inference_mean_layer_call_and_return_conditional_losses_13521f��0�-
&�#
!�
inputs����������
� ",�)
"�
tensor_0���������

� �
$__inference_mean_layer_call_fn_13511[��0�-
&�#
!�
inputs����������
� "!�
unknown���������
�
#__inference_signature_wrapper_12481�9%&/012?@IJKLYZcdefst}~�����������������K�H
� 
A�>
<
input_layer-�*
input_layer���������"W�T
*
logVar �
logvar���������

&
mean�
mean���������
