Щ 
▓Б
^
AssignVariableOp
resource
value"dtype"
dtypetype"
validate_shapebool( И
А
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
Ы
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
resourceИ
√
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
epsilonfloat%╖╤8"&
exponential_avg_factorfloat%  А?";
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
alphafloat%═╠L>"
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
Ж
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool("
allow_missing_filesbool( И
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
dtypetypeИ
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
list(type)(0И
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0И
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
┴
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
executor_typestring Ии
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
Ц
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 И"serve*2.12.02v2.12.0-rc1-12-g0db597d0d758ТБ
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
shape:	└
*
shared_namelogVar/kernel
p
!logVar/kernel/Read/ReadVariableOpReadVariableOplogVar/kernel*
_output_shapes
:	└
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
shape:	└
*
shared_namemean/kernel
l
mean/kernel/Read/ReadVariableOpReadVariableOpmean/kernel*
_output_shapes
:	└
*
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
А
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
А
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
А
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
А
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
О
serving_default_input_layerPlaceholder*/
_output_shapes
:         *
dtype0*$
shape:         
Щ
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_layerconv2D1/kernelconv2D1/bias	bn1/gammabn1/betabn1/moving_meanbn1/moving_varianceconv2D2/kernelconv2D2/bias	bn2/gammabn2/betabn2/moving_meanbn2/moving_varianceconv2D3/kernelconv2D3/bias	bn3/gammabn3/betabn3/moving_meanbn3/moving_varianceconv2D4/kernelconv2D4/bias	bn4/gammabn4/betabn4/moving_meanbn4/moving_variancelogVar/kernellogVar/biasmean/kernel	mean/bias*(
Tin!
2*
Tout
2*
_collective_manager_ids
 *:
_output_shapes(
&:         
:         
*>
_read_only_resource_inputs 
	
*-
config_proto

CPU

GPU 2J 8В *,
f'R%
#__inference_signature_wrapper_11111

NoOpNoOp
·g
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*╡g
valueлgBиg Bбg
Ч
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
layer-13
layer_with_weights-8
layer-14
layer_with_weights-9
layer-15
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
_default_save_signature

signatures*
* 
╚
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses

kernel
 bias
 !_jit_compiled_convolution_op*
╒
"	variables
#trainable_variables
$regularization_losses
%	keras_api
&__call__
*'&call_and_return_all_conditional_losses
(axis
	)gamma
*beta
+moving_mean
,moving_variance*
О
-	variables
.trainable_variables
/regularization_losses
0	keras_api
1__call__
*2&call_and_return_all_conditional_losses* 
╚
3	variables
4trainable_variables
5regularization_losses
6	keras_api
7__call__
*8&call_and_return_all_conditional_losses

9kernel
:bias
 ;_jit_compiled_convolution_op*
╒
<	variables
=trainable_variables
>regularization_losses
?	keras_api
@__call__
*A&call_and_return_all_conditional_losses
Baxis
	Cgamma
Dbeta
Emoving_mean
Fmoving_variance*
О
G	variables
Htrainable_variables
Iregularization_losses
J	keras_api
K__call__
*L&call_and_return_all_conditional_losses* 
╚
M	variables
Ntrainable_variables
Oregularization_losses
P	keras_api
Q__call__
*R&call_and_return_all_conditional_losses

Skernel
Tbias
 U_jit_compiled_convolution_op*
╒
V	variables
Wtrainable_variables
Xregularization_losses
Y	keras_api
Z__call__
*[&call_and_return_all_conditional_losses
\axis
	]gamma
^beta
_moving_mean
`moving_variance*
О
a	variables
btrainable_variables
cregularization_losses
d	keras_api
e__call__
*f&call_and_return_all_conditional_losses* 
╚
g	variables
htrainable_variables
iregularization_losses
j	keras_api
k__call__
*l&call_and_return_all_conditional_losses

mkernel
nbias
 o_jit_compiled_convolution_op*
╒
p	variables
qtrainable_variables
rregularization_losses
s	keras_api
t__call__
*u&call_and_return_all_conditional_losses
vaxis
	wgamma
xbeta
ymoving_mean
zmoving_variance*
П
{	variables
|trainable_variables
}regularization_losses
~	keras_api
__call__
+А&call_and_return_all_conditional_losses* 
Ф
Б	variables
Вtrainable_variables
Гregularization_losses
Д	keras_api
Е__call__
+Ж&call_and_return_all_conditional_losses* 
о
З	variables
Иtrainable_variables
Йregularization_losses
К	keras_api
Л__call__
+М&call_and_return_all_conditional_losses
Нkernel
	Оbias*
о
П	variables
Рtrainable_variables
Сregularization_losses
Т	keras_api
У__call__
+Ф&call_and_return_all_conditional_losses
Хkernel
	Цbias*
▐
0
 1
)2
*3
+4
,5
96
:7
C8
D9
E10
F11
S12
T13
]14
^15
_16
`17
m18
n19
w20
x21
y22
z23
Н24
О25
Х26
Ц27*
Ю
0
 1
)2
*3
94
:5
C6
D7
S8
T9
]10
^11
m12
n13
w14
x15
Н16
О17
Х18
Ц19*
* 
╡
Чnon_trainable_variables
Шlayers
Щmetrics
 Ъlayer_regularization_losses
Ыlayer_metrics
	variables
trainable_variables
regularization_losses
__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*
:
Ьtrace_0
Эtrace_1
Юtrace_2
Яtrace_3* 
:
аtrace_0
бtrace_1
вtrace_2
гtrace_3* 
* 

дserving_default* 

0
 1*

0
 1*
* 
Ш
еnon_trainable_variables
жlayers
зmetrics
 иlayer_regularization_losses
йlayer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*

кtrace_0* 

лtrace_0* 
^X
VARIABLE_VALUEconv2D1/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE*
ZT
VARIABLE_VALUEconv2D1/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 
 
)0
*1
+2
,3*

)0
*1*
* 
Ш
мnon_trainable_variables
нlayers
оmetrics
 пlayer_regularization_losses
░layer_metrics
"	variables
#trainable_variables
$regularization_losses
&__call__
*'&call_and_return_all_conditional_losses
&'"call_and_return_conditional_losses*

▒trace_0
▓trace_1* 

│trace_0
┤trace_1* 
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
Ц
╡non_trainable_variables
╢layers
╖metrics
 ╕layer_regularization_losses
╣layer_metrics
-	variables
.trainable_variables
/regularization_losses
1__call__
*2&call_and_return_all_conditional_losses
&2"call_and_return_conditional_losses* 

║trace_0* 

╗trace_0* 

90
:1*

90
:1*
* 
Ш
╝non_trainable_variables
╜layers
╛metrics
 ┐layer_regularization_losses
└layer_metrics
3	variables
4trainable_variables
5regularization_losses
7__call__
*8&call_and_return_all_conditional_losses
&8"call_and_return_conditional_losses*

┴trace_0* 

┬trace_0* 
^X
VARIABLE_VALUEconv2D2/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE*
ZT
VARIABLE_VALUEconv2D2/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 
 
C0
D1
E2
F3*

C0
D1*
* 
Ш
├non_trainable_variables
─layers
┼metrics
 ╞layer_regularization_losses
╟layer_metrics
<	variables
=trainable_variables
>regularization_losses
@__call__
*A&call_and_return_all_conditional_losses
&A"call_and_return_conditional_losses*

╚trace_0
╔trace_1* 

╩trace_0
╦trace_1* 
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
Ц
╠non_trainable_variables
═layers
╬metrics
 ╧layer_regularization_losses
╨layer_metrics
G	variables
Htrainable_variables
Iregularization_losses
K__call__
*L&call_and_return_all_conditional_losses
&L"call_and_return_conditional_losses* 

╤trace_0* 

╥trace_0* 

S0
T1*

S0
T1*
* 
Ш
╙non_trainable_variables
╘layers
╒metrics
 ╓layer_regularization_losses
╫layer_metrics
M	variables
Ntrainable_variables
Oregularization_losses
Q__call__
*R&call_and_return_all_conditional_losses
&R"call_and_return_conditional_losses*

╪trace_0* 

┘trace_0* 
^X
VARIABLE_VALUEconv2D3/kernel6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUE*
ZT
VARIABLE_VALUEconv2D3/bias4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 
 
]0
^1
_2
`3*

]0
^1*
* 
Ш
┌non_trainable_variables
█layers
▄metrics
 ▌layer_regularization_losses
▐layer_metrics
V	variables
Wtrainable_variables
Xregularization_losses
Z__call__
*[&call_and_return_all_conditional_losses
&["call_and_return_conditional_losses*

▀trace_0
рtrace_1* 

сtrace_0
тtrace_1* 
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
Ц
уnon_trainable_variables
фlayers
хmetrics
 цlayer_regularization_losses
чlayer_metrics
a	variables
btrainable_variables
cregularization_losses
e__call__
*f&call_and_return_all_conditional_losses
&f"call_and_return_conditional_losses* 

шtrace_0* 

щtrace_0* 

m0
n1*

m0
n1*
* 
Ш
ъnon_trainable_variables
ыlayers
ьmetrics
 эlayer_regularization_losses
юlayer_metrics
g	variables
htrainable_variables
iregularization_losses
k__call__
*l&call_and_return_all_conditional_losses
&l"call_and_return_conditional_losses*

яtrace_0* 

Ёtrace_0* 
^X
VARIABLE_VALUEconv2D4/kernel6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUE*
ZT
VARIABLE_VALUEconv2D4/bias4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 
 
w0
x1
y2
z3*

w0
x1*
* 
Ш
ёnon_trainable_variables
Єlayers
єmetrics
 Їlayer_regularization_losses
їlayer_metrics
p	variables
qtrainable_variables
rregularization_losses
t__call__
*u&call_and_return_all_conditional_losses
&u"call_and_return_conditional_losses*

Ўtrace_0
ўtrace_1* 

°trace_0
∙trace_1* 
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
Ш
·non_trainable_variables
√layers
№metrics
 ¤layer_regularization_losses
■layer_metrics
{	variables
|trainable_variables
}regularization_losses
__call__
+А&call_and_return_all_conditional_losses
'А"call_and_return_conditional_losses* 

 trace_0* 

Аtrace_0* 
* 
* 
* 
Ь
Бnon_trainable_variables
Вlayers
Гmetrics
 Дlayer_regularization_losses
Еlayer_metrics
Б	variables
Вtrainable_variables
Гregularization_losses
Е__call__
+Ж&call_and_return_all_conditional_losses
'Ж"call_and_return_conditional_losses* 

Жtrace_0* 

Зtrace_0* 

Н0
О1*

Н0
О1*
* 
Ю
Иnon_trainable_variables
Йlayers
Кmetrics
 Лlayer_regularization_losses
Мlayer_metrics
З	variables
Иtrainable_variables
Йregularization_losses
Л__call__
+М&call_and_return_all_conditional_losses
'М"call_and_return_conditional_losses*

Нtrace_0* 

Оtrace_0* 
[U
VARIABLE_VALUEmean/kernel6layer_with_weights-8/kernel/.ATTRIBUTES/VARIABLE_VALUE*
WQ
VARIABLE_VALUE	mean/bias4layer_with_weights-8/bias/.ATTRIBUTES/VARIABLE_VALUE*

Х0
Ц1*

Х0
Ц1*
* 
Ю
Пnon_trainable_variables
Рlayers
Сmetrics
 Тlayer_regularization_losses
Уlayer_metrics
П	variables
Рtrainable_variables
Сregularization_losses
У__call__
+Ф&call_and_return_all_conditional_losses
'Ф"call_and_return_conditional_losses*

Фtrace_0* 

Хtrace_0* 
]W
VARIABLE_VALUElogVar/kernel6layer_with_weights-9/kernel/.ATTRIBUTES/VARIABLE_VALUE*
YS
VARIABLE_VALUElogVar/bias4layer_with_weights-9/bias/.ATTRIBUTES/VARIABLE_VALUE*
<
+0
,1
E2
F3
_4
`5
y6
z7*
z
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
15*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
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
+0
,1*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
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
E0
F1*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
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
_0
`1*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
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
y0
z1*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
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
╬
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filenameconv2D1/kernelconv2D1/bias	bn1/gammabn1/betabn1/moving_meanbn1/moving_varianceconv2D2/kernelconv2D2/bias	bn2/gammabn2/betabn2/moving_meanbn2/moving_varianceconv2D3/kernelconv2D3/bias	bn3/gammabn3/betabn3/moving_meanbn3/moving_varianceconv2D4/kernelconv2D4/bias	bn4/gammabn4/betabn4/moving_meanbn4/moving_variancemean/kernel	mean/biaslogVar/kernellogVar/biasConst*)
Tin"
 2*
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
GPU 2J 8В *'
f"R 
__inference__traced_save_12048
╔
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenameconv2D1/kernelconv2D1/bias	bn1/gammabn1/betabn1/moving_meanbn1/moving_varianceconv2D2/kernelconv2D2/bias	bn2/gammabn2/betabn2/moving_meanbn2/moving_varianceconv2D3/kernelconv2D3/bias	bn3/gammabn3/betabn3/moving_meanbn3/moving_varianceconv2D4/kernelconv2D4/bias	bn4/gammabn4/betabn4/moving_meanbn4/moving_variancemean/kernel	mean/biaslogVar/kernellogVar/bias*(
Tin!
2*
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
GPU 2J 8В **
f%R#
!__inference__traced_restore_12142¤╙
є
н
>__inference_bn3_layer_call_and_return_conditional_losses_10192

inputs%
readvariableop_resource:@'
readvariableop_1_resource:@6
(fusedbatchnormv3_readvariableop_resource:@8
*fusedbatchnormv3_readvariableop_1_resource:@
identityИвAssignNewValueвAssignNewValue_1вFusedBatchNormV3/ReadVariableOpв!FusedBatchNormV3/ReadVariableOp_1вReadVariableOpвReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype0Д
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype0И
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype0╓
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+                           @:@:@:@:@:*
epsilon%oГ:*
exponential_avg_factor%
╫#<╞
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(╨
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(}
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+                           @╘
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+                           @: : : : 2$
AssignNewValue_1AssignNewValue_12 
AssignNewValueAssignNewValue2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2$
ReadVariableOp_1ReadVariableOp_12 
ReadVariableOpReadVariableOp:i e
A
_output_shapes/
-:+                           @
 
_user_specified_nameinputs
─
^
B__inference_flatten_layer_call_and_return_conditional_losses_10439

inputs
identityV
ConstConst*
_output_shapes
:*
dtype0*
valueB"    @  ]
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:         └Y
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:         └"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:         @:W S
/
_output_shapes
:         @
 
_user_specified_nameinputs
щ
╛
#__inference_bn1_layer_call_fn_11475

inputs
unknown: 
	unknown_0: 
	unknown_1: 
	unknown_2: 
identityИвStatefulPartitionedCallЕ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+                            *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *G
fBR@
>__inference_bn1_layer_call_and_return_conditional_losses_10064Й
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+                            `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+                            : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+                            
 
_user_specified_nameinputs
╣╠
╞
__inference__traced_save_12048
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
-read_23_disablecopyonread_bn4_moving_variance:@8
%read_24_disablecopyonread_mean_kernel:	└
1
#read_25_disablecopyonread_mean_bias:
:
'read_26_disablecopyonread_logvar_kernel:	└
3
%read_27_disablecopyonread_logvar_bias:

savev2_const
identity_57ИвMergeV2CheckpointsвRead/DisableCopyOnReadвRead/ReadVariableOpвRead_1/DisableCopyOnReadвRead_1/ReadVariableOpвRead_10/DisableCopyOnReadвRead_10/ReadVariableOpвRead_11/DisableCopyOnReadвRead_11/ReadVariableOpвRead_12/DisableCopyOnReadвRead_12/ReadVariableOpвRead_13/DisableCopyOnReadвRead_13/ReadVariableOpвRead_14/DisableCopyOnReadвRead_14/ReadVariableOpвRead_15/DisableCopyOnReadвRead_15/ReadVariableOpвRead_16/DisableCopyOnReadвRead_16/ReadVariableOpвRead_17/DisableCopyOnReadвRead_17/ReadVariableOpвRead_18/DisableCopyOnReadвRead_18/ReadVariableOpвRead_19/DisableCopyOnReadвRead_19/ReadVariableOpвRead_2/DisableCopyOnReadвRead_2/ReadVariableOpвRead_20/DisableCopyOnReadвRead_20/ReadVariableOpвRead_21/DisableCopyOnReadвRead_21/ReadVariableOpвRead_22/DisableCopyOnReadвRead_22/ReadVariableOpвRead_23/DisableCopyOnReadвRead_23/ReadVariableOpвRead_24/DisableCopyOnReadвRead_24/ReadVariableOpвRead_25/DisableCopyOnReadвRead_25/ReadVariableOpвRead_26/DisableCopyOnReadвRead_26/ReadVariableOpвRead_27/DisableCopyOnReadвRead_27/ReadVariableOpвRead_3/DisableCopyOnReadвRead_3/ReadVariableOpвRead_4/DisableCopyOnReadвRead_4/ReadVariableOpвRead_5/DisableCopyOnReadвRead_5/ReadVariableOpвRead_6/DisableCopyOnReadвRead_6/ReadVariableOpвRead_7/DisableCopyOnReadвRead_7/ReadVariableOpвRead_8/DisableCopyOnReadвRead_8/ReadVariableOpвRead_9/DisableCopyOnReadвRead_9/ReadVariableOpw
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
_temp/partБ
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
value	B : У
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: w
Read/DisableCopyOnReadDisableCopyOnRead%read_disablecopyonread_conv2d1_kernel"/device:CPU:0*
_output_shapes
 й
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
 б
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
 Ю
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
 Э
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
 д
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
: А
Read_5/DisableCopyOnReadDisableCopyOnRead,read_5_disablecopyonread_bn1_moving_variance"/device:CPU:0*
_output_shapes
 и
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
 п
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
 б
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
 Ю
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
 Э
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
 з
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
:@В
Read_11/DisableCopyOnReadDisableCopyOnRead-read_11_disablecopyonread_bn2_moving_variance"/device:CPU:0*
_output_shapes
 л
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
 ▓
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
 д
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
 б
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
 а
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
 з
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
:@В
Read_17/DisableCopyOnReadDisableCopyOnRead-read_17_disablecopyonread_bn3_moving_variance"/device:CPU:0*
_output_shapes
 л
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
 ▓
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
 д
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
 б
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
 а
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
 з
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
:@В
Read_23/DisableCopyOnReadDisableCopyOnRead-read_23_disablecopyonread_bn4_moving_variance"/device:CPU:0*
_output_shapes
 л
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
:@z
Read_24/DisableCopyOnReadDisableCopyOnRead%read_24_disablecopyonread_mean_kernel"/device:CPU:0*
_output_shapes
 и
Read_24/ReadVariableOpReadVariableOp%read_24_disablecopyonread_mean_kernel^Read_24/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:	└
*
dtype0p
Identity_48IdentityRead_24/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:	└
f
Identity_49IdentityIdentity_48:output:0"/device:CPU:0*
T0*
_output_shapes
:	└
x
Read_25/DisableCopyOnReadDisableCopyOnRead#read_25_disablecopyonread_mean_bias"/device:CPU:0*
_output_shapes
 б
Read_25/ReadVariableOpReadVariableOp#read_25_disablecopyonread_mean_bias^Read_25/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:
*
dtype0k
Identity_50IdentityRead_25/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:
a
Identity_51IdentityIdentity_50:output:0"/device:CPU:0*
T0*
_output_shapes
:
|
Read_26/DisableCopyOnReadDisableCopyOnRead'read_26_disablecopyonread_logvar_kernel"/device:CPU:0*
_output_shapes
 к
Read_26/ReadVariableOpReadVariableOp'read_26_disablecopyonread_logvar_kernel^Read_26/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:	└
*
dtype0p
Identity_52IdentityRead_26/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:	└
f
Identity_53IdentityIdentity_52:output:0"/device:CPU:0*
T0*
_output_shapes
:	└
z
Read_27/DisableCopyOnReadDisableCopyOnRead%read_27_disablecopyonread_logvar_bias"/device:CPU:0*
_output_shapes
 г
Read_27/ReadVariableOpReadVariableOp%read_27_disablecopyonread_logvar_bias^Read_27/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:
*
dtype0k
Identity_54IdentityRead_27/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:
a
Identity_55IdentityIdentity_54:output:0"/device:CPU:0*
T0*
_output_shapes
:
╩
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*є
valueщBцB6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-1/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-1/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-1/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-3/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-3/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-3/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-5/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-5/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-5/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-7/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-7/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-7/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-7/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-8/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-8/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-9/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-9/bias/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPHз
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*M
valueDBBB B B B B B B B B B B B B B B B B B B B B B B B B B B B B ╫
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0Identity_1:output:0Identity_3:output:0Identity_5:output:0Identity_7:output:0Identity_9:output:0Identity_11:output:0Identity_13:output:0Identity_15:output:0Identity_17:output:0Identity_19:output:0Identity_21:output:0Identity_23:output:0Identity_25:output:0Identity_27:output:0Identity_29:output:0Identity_31:output:0Identity_33:output:0Identity_35:output:0Identity_37:output:0Identity_39:output:0Identity_41:output:0Identity_43:output:0Identity_45:output:0Identity_47:output:0Identity_49:output:0Identity_51:output:0Identity_53:output:0Identity_55:output:0savev2_const"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *+
dtypes!
2Р
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:│
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 i
Identity_56Identityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: U
Identity_57IdentityIdentity_56:output:0^NoOp*
T0*
_output_shapes
: П
NoOpNoOp^MergeV2Checkpoints^Read/DisableCopyOnRead^Read/ReadVariableOp^Read_1/DisableCopyOnRead^Read_1/ReadVariableOp^Read_10/DisableCopyOnRead^Read_10/ReadVariableOp^Read_11/DisableCopyOnRead^Read_11/ReadVariableOp^Read_12/DisableCopyOnRead^Read_12/ReadVariableOp^Read_13/DisableCopyOnRead^Read_13/ReadVariableOp^Read_14/DisableCopyOnRead^Read_14/ReadVariableOp^Read_15/DisableCopyOnRead^Read_15/ReadVariableOp^Read_16/DisableCopyOnRead^Read_16/ReadVariableOp^Read_17/DisableCopyOnRead^Read_17/ReadVariableOp^Read_18/DisableCopyOnRead^Read_18/ReadVariableOp^Read_19/DisableCopyOnRead^Read_19/ReadVariableOp^Read_2/DisableCopyOnRead^Read_2/ReadVariableOp^Read_20/DisableCopyOnRead^Read_20/ReadVariableOp^Read_21/DisableCopyOnRead^Read_21/ReadVariableOp^Read_22/DisableCopyOnRead^Read_22/ReadVariableOp^Read_23/DisableCopyOnRead^Read_23/ReadVariableOp^Read_24/DisableCopyOnRead^Read_24/ReadVariableOp^Read_25/DisableCopyOnRead^Read_25/ReadVariableOp^Read_26/DisableCopyOnRead^Read_26/ReadVariableOp^Read_27/DisableCopyOnRead^Read_27/ReadVariableOp^Read_3/DisableCopyOnRead^Read_3/ReadVariableOp^Read_4/DisableCopyOnRead^Read_4/ReadVariableOp^Read_5/DisableCopyOnRead^Read_5/ReadVariableOp^Read_6/DisableCopyOnRead^Read_6/ReadVariableOp^Read_7/DisableCopyOnRead^Read_7/ReadVariableOp^Read_8/DisableCopyOnRead^Read_8/ReadVariableOp^Read_9/DisableCopyOnRead^Read_9/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "#
identity_57Identity_57:output:0*O
_input_shapes>
<: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2(
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
Read_27/ReadVariableOpRead_27/ReadVariableOp24
Read_3/DisableCopyOnReadRead_3/DisableCopyOnRead2.
Read_3/ReadVariableOpRead_3/ReadVariableOp24
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
Read_9/ReadVariableOpRead_9/ReadVariableOp:

_output_shapes
: :C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
╣
Й
>__inference_bn4_layer_call_and_return_conditional_losses_11797

inputs%
readvariableop_resource:@'
readvariableop_1_resource:@6
(fusedbatchnormv3_readvariableop_resource:@8
*fusedbatchnormv3_readvariableop_1_resource:@
identityИвFusedBatchNormV3/ReadVariableOpв!FusedBatchNormV3/ReadVariableOp_1вReadVariableOpвReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype0Д
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype0И
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype0╚
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+                           @:@:@:@:@:*
epsilon%oГ:*
is_training( }
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+                           @░
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+                           @: : : : 2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2$
ReadVariableOp_1ReadVariableOp_12 
ReadVariableOpReadVariableOp:i e
A
_output_shapes/
-:+                           @
 
_user_specified_nameinputs
є
н
>__inference_bn2_layer_call_and_return_conditional_losses_10128

inputs%
readvariableop_resource:@'
readvariableop_1_resource:@6
(fusedbatchnormv3_readvariableop_resource:@8
*fusedbatchnormv3_readvariableop_1_resource:@
identityИвAssignNewValueвAssignNewValue_1вFusedBatchNormV3/ReadVariableOpв!FusedBatchNormV3/ReadVariableOp_1вReadVariableOpвReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype0Д
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype0И
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype0╓
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+                           @:@:@:@:@:*
epsilon%oГ:*
exponential_avg_factor%
╫#<╞
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(╨
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(}
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+                           @╘
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+                           @: : : : 2$
AssignNewValue_1AssignNewValue_12 
AssignNewValueAssignNewValue2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2$
ReadVariableOp_1ReadVariableOp_12 
ReadVariableOpReadVariableOp:i e
A
_output_shapes/
-:+                           @
 
_user_specified_nameinputs
е

√
B__inference_conv2D4_layer_call_and_return_conditional_losses_11735

inputs8
conv2d_readvariableop_resource:@@-
biasadd_readvariableop_resource:@
identityИвBiasAdd/ReadVariableOpвConv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype0Щ
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         @*
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
:         @g
IdentityIdentityBiasAdd:output:0^NoOp*
T0*/
_output_shapes
:         @w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:         @: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:         @
 
_user_specified_nameinputs
√
]
A__inference_lReLU1_layer_call_and_return_conditional_losses_10335

inputs
identity_
	LeakyRelu	LeakyReluinputs*/
_output_shapes
:          *
alpha%ЪЩЩ>g
IdentityIdentityLeakyRelu:activations:0*
T0*/
_output_shapes
:          "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:          :W S
/
_output_shapes
:          
 
_user_specified_nameinputs
є
н
>__inference_bn1_layer_call_and_return_conditional_losses_10064

inputs%
readvariableop_resource: '
readvariableop_1_resource: 6
(fusedbatchnormv3_readvariableop_resource: 8
*fusedbatchnormv3_readvariableop_1_resource: 
identityИвAssignNewValueвAssignNewValue_1вFusedBatchNormV3/ReadVariableOpв!FusedBatchNormV3/ReadVariableOp_1вReadVariableOpвReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype0Д
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype0И
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype0╓
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+                            : : : : :*
epsilon%oГ:*
exponential_avg_factor%
╫#<╞
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(╨
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(}
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+                            ╘
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+                            : : : : 2$
AssignNewValue_1AssignNewValue_12 
AssignNewValueAssignNewValue2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2$
ReadVariableOp_1ReadVariableOp_12 
ReadVariableOpReadVariableOp:i e
A
_output_shapes/
-:+                            
 
_user_specified_nameinputs
▌v
в
!__inference__traced_restore_12142
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
'assignvariableop_23_bn4_moving_variance:@2
assignvariableop_24_mean_kernel:	└
+
assignvariableop_25_mean_bias:
4
!assignvariableop_26_logvar_kernel:	└
-
assignvariableop_27_logvar_bias:

identity_29ИвAssignVariableOpвAssignVariableOp_1вAssignVariableOp_10вAssignVariableOp_11вAssignVariableOp_12вAssignVariableOp_13вAssignVariableOp_14вAssignVariableOp_15вAssignVariableOp_16вAssignVariableOp_17вAssignVariableOp_18вAssignVariableOp_19вAssignVariableOp_2вAssignVariableOp_20вAssignVariableOp_21вAssignVariableOp_22вAssignVariableOp_23вAssignVariableOp_24вAssignVariableOp_25вAssignVariableOp_26вAssignVariableOp_27вAssignVariableOp_3вAssignVariableOp_4вAssignVariableOp_5вAssignVariableOp_6вAssignVariableOp_7вAssignVariableOp_8вAssignVariableOp_9═
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*є
valueщBцB6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-1/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-1/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-1/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-3/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-3/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-3/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-5/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-5/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-5/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-7/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-7/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-7/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-7/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-8/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-8/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-9/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-9/bias/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPHк
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*M
valueDBBB B B B B B B B B B B B B B B B B B B B B B B B B B B B B ░
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*И
_output_shapesv
t:::::::::::::::::::::::::::::*+
dtypes!
2[
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:▓
AssignVariableOpAssignVariableOpassignvariableop_conv2d1_kernelIdentity:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:╢
AssignVariableOp_1AssignVariableOpassignvariableop_1_conv2d1_biasIdentity_1:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:│
AssignVariableOp_2AssignVariableOpassignvariableop_2_bn1_gammaIdentity_2:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:▓
AssignVariableOp_3AssignVariableOpassignvariableop_3_bn1_betaIdentity_3:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:╣
AssignVariableOp_4AssignVariableOp"assignvariableop_4_bn1_moving_meanIdentity_4:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:╜
AssignVariableOp_5AssignVariableOp&assignvariableop_5_bn1_moving_varianceIdentity_5:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:╕
AssignVariableOp_6AssignVariableOp!assignvariableop_6_conv2d2_kernelIdentity_6:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:╢
AssignVariableOp_7AssignVariableOpassignvariableop_7_conv2d2_biasIdentity_7:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:│
AssignVariableOp_8AssignVariableOpassignvariableop_8_bn2_gammaIdentity_8:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:▓
AssignVariableOp_9AssignVariableOpassignvariableop_9_bn2_betaIdentity_9:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:╝
AssignVariableOp_10AssignVariableOp#assignvariableop_10_bn2_moving_meanIdentity_10:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:└
AssignVariableOp_11AssignVariableOp'assignvariableop_11_bn2_moving_varianceIdentity_11:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:╗
AssignVariableOp_12AssignVariableOp"assignvariableop_12_conv2d3_kernelIdentity_12:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:╣
AssignVariableOp_13AssignVariableOp assignvariableop_13_conv2d3_biasIdentity_13:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:╢
AssignVariableOp_14AssignVariableOpassignvariableop_14_bn3_gammaIdentity_14:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:╡
AssignVariableOp_15AssignVariableOpassignvariableop_15_bn3_betaIdentity_15:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:╝
AssignVariableOp_16AssignVariableOp#assignvariableop_16_bn3_moving_meanIdentity_16:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:└
AssignVariableOp_17AssignVariableOp'assignvariableop_17_bn3_moving_varianceIdentity_17:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:╗
AssignVariableOp_18AssignVariableOp"assignvariableop_18_conv2d4_kernelIdentity_18:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:╣
AssignVariableOp_19AssignVariableOp assignvariableop_19_conv2d4_biasIdentity_19:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:╢
AssignVariableOp_20AssignVariableOpassignvariableop_20_bn4_gammaIdentity_20:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:╡
AssignVariableOp_21AssignVariableOpassignvariableop_21_bn4_betaIdentity_21:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:╝
AssignVariableOp_22AssignVariableOp#assignvariableop_22_bn4_moving_meanIdentity_22:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:└
AssignVariableOp_23AssignVariableOp'assignvariableop_23_bn4_moving_varianceIdentity_23:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:╕
AssignVariableOp_24AssignVariableOpassignvariableop_24_mean_kernelIdentity_24:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:╢
AssignVariableOp_25AssignVariableOpassignvariableop_25_mean_biasIdentity_25:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:║
AssignVariableOp_26AssignVariableOp!assignvariableop_26_logvar_kernelIdentity_26:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:╕
AssignVariableOp_27AssignVariableOpassignvariableop_27_logvar_biasIdentity_27:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0Y
NoOpNoOp"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 ╖
Identity_28Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: W
Identity_29IdentityIdentity_28:output:0^NoOp_1*
T0*
_output_shapes
: д
NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*"
_acd_function_control_output(*
_output_shapes
 "#
identity_29Identity_29:output:0*M
_input_shapes<
:: : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2*
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
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_20AssignVariableOp_202*
AssignVariableOp_21AssignVariableOp_212*
AssignVariableOp_22AssignVariableOp_222*
AssignVariableOp_23AssignVariableOp_232*
AssignVariableOp_24AssignVariableOp_242*
AssignVariableOp_25AssignVariableOp_252*
AssignVariableOp_26AssignVariableOp_262*
AssignVariableOp_27AssignVariableOp_272(
AssignVariableOp_2AssignVariableOp_22(
AssignVariableOp_3AssignVariableOp_32(
AssignVariableOp_4AssignVariableOp_42(
AssignVariableOp_5AssignVariableOp_52(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_92$
AssignVariableOpAssignVariableOp:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
╣
Й
>__inference_bn3_layer_call_and_return_conditional_losses_10210

inputs%
readvariableop_resource:@'
readvariableop_1_resource:@6
(fusedbatchnormv3_readvariableop_resource:@8
*fusedbatchnormv3_readvariableop_1_resource:@
identityИвFusedBatchNormV3/ReadVariableOpв!FusedBatchNormV3/ReadVariableOp_1вReadVariableOpвReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype0Д
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype0И
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype0╚
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+                           @:@:@:@:@:*
epsilon%oГ:*
is_training( }
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+                           @░
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+                           @: : : : 2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2$
ReadVariableOp_1ReadVariableOp_12 
ReadVariableOpReadVariableOp:i e
A
_output_shapes/
-:+                           @
 
_user_specified_nameinputs
є
н
>__inference_bn3_layer_call_and_return_conditional_losses_11688

inputs%
readvariableop_resource:@'
readvariableop_1_resource:@6
(fusedbatchnormv3_readvariableop_resource:@8
*fusedbatchnormv3_readvariableop_1_resource:@
identityИвAssignNewValueвAssignNewValue_1вFusedBatchNormV3/ReadVariableOpв!FusedBatchNormV3/ReadVariableOp_1вReadVariableOpвReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype0Д
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype0И
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype0╓
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+                           @:@:@:@:@:*
epsilon%oГ:*
exponential_avg_factor%
╫#<╞
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(╨
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(}
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+                           @╘
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+                           @: : : : 2$
AssignNewValue_1AssignNewValue_12 
AssignNewValueAssignNewValue2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2$
ReadVariableOp_1ReadVariableOp_12 
ReadVariableOpReadVariableOp:i e
A
_output_shapes/
-:+                           @
 
_user_specified_nameinputs
╣
Й
>__inference_bn3_layer_call_and_return_conditional_losses_11706

inputs%
readvariableop_resource:@'
readvariableop_1_resource:@6
(fusedbatchnormv3_readvariableop_resource:@8
*fusedbatchnormv3_readvariableop_1_resource:@
identityИвFusedBatchNormV3/ReadVariableOpв!FusedBatchNormV3/ReadVariableOp_1вReadVariableOpвReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype0Д
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype0И
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype0╚
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+                           @:@:@:@:@:*
epsilon%oГ:*
is_training( }
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+                           @░
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+                           @: : : : 2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2$
ReadVariableOp_1ReadVariableOp_12 
ReadVariableOpReadVariableOp:i e
A
_output_shapes/
-:+                           @
 
_user_specified_nameinputs
р|
П
B__inference_encoder_layer_call_and_return_conditional_losses_11340

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
.bn4_fusedbatchnormv3_readvariableop_1_resource:@8
%logvar_matmul_readvariableop_resource:	└
4
&logvar_biasadd_readvariableop_resource:
6
#mean_matmul_readvariableop_resource:	└
2
$mean_biasadd_readvariableop_resource:

identity

identity_1Ивbn1/AssignNewValueвbn1/AssignNewValue_1в#bn1/FusedBatchNormV3/ReadVariableOpв%bn1/FusedBatchNormV3/ReadVariableOp_1вbn1/ReadVariableOpвbn1/ReadVariableOp_1вbn2/AssignNewValueвbn2/AssignNewValue_1в#bn2/FusedBatchNormV3/ReadVariableOpв%bn2/FusedBatchNormV3/ReadVariableOp_1вbn2/ReadVariableOpвbn2/ReadVariableOp_1вbn3/AssignNewValueвbn3/AssignNewValue_1в#bn3/FusedBatchNormV3/ReadVariableOpв%bn3/FusedBatchNormV3/ReadVariableOp_1вbn3/ReadVariableOpвbn3/ReadVariableOp_1вbn4/AssignNewValueвbn4/AssignNewValue_1в#bn4/FusedBatchNormV3/ReadVariableOpв%bn4/FusedBatchNormV3/ReadVariableOp_1вbn4/ReadVariableOpвbn4/ReadVariableOp_1вconv2D1/BiasAdd/ReadVariableOpвconv2D1/Conv2D/ReadVariableOpвconv2D2/BiasAdd/ReadVariableOpвconv2D2/Conv2D/ReadVariableOpвconv2D3/BiasAdd/ReadVariableOpвconv2D3/Conv2D/ReadVariableOpвconv2D4/BiasAdd/ReadVariableOpвconv2D4/Conv2D/ReadVariableOpвlogVar/BiasAdd/ReadVariableOpвlogVar/MatMul/ReadVariableOpвmean/BiasAdd/ReadVariableOpвmean/MatMul/ReadVariableOpМ
conv2D1/Conv2D/ReadVariableOpReadVariableOp&conv2d1_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype0й
conv2D1/Conv2DConv2Dinputs%conv2D1/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:          *
paddingSAME*
strides
В
conv2D1/BiasAdd/ReadVariableOpReadVariableOp'conv2d1_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0Х
conv2D1/BiasAddBiasAddconv2D1/Conv2D:output:0&conv2D1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:          j
bn1/ReadVariableOpReadVariableOpbn1_readvariableop_resource*
_output_shapes
: *
dtype0n
bn1/ReadVariableOp_1ReadVariableOpbn1_readvariableop_1_resource*
_output_shapes
: *
dtype0М
#bn1/FusedBatchNormV3/ReadVariableOpReadVariableOp,bn1_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype0Р
%bn1/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp.bn1_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype0ъ
bn1/FusedBatchNormV3FusedBatchNormV3conv2D1/BiasAdd:output:0bn1/ReadVariableOp:value:0bn1/ReadVariableOp_1:value:0+bn1/FusedBatchNormV3/ReadVariableOp:value:0-bn1/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:          : : : : :*
epsilon%oГ:*
exponential_avg_factor%
╫#<╓
bn1/AssignNewValueAssignVariableOp,bn1_fusedbatchnormv3_readvariableop_resource!bn1/FusedBatchNormV3:batch_mean:0$^bn1/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(р
bn1/AssignNewValue_1AssignVariableOp.bn1_fusedbatchnormv3_readvariableop_1_resource%bn1/FusedBatchNormV3:batch_variance:0&^bn1/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(x
lReLU1/LeakyRelu	LeakyRelubn1/FusedBatchNormV3:y:0*/
_output_shapes
:          *
alpha%ЪЩЩ>М
conv2D2/Conv2D/ReadVariableOpReadVariableOp&conv2d2_conv2d_readvariableop_resource*&
_output_shapes
: @*
dtype0┴
conv2D2/Conv2DConv2DlReLU1/LeakyRelu:activations:0%conv2D2/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         @*
paddingSAME*
strides
В
conv2D2/BiasAdd/ReadVariableOpReadVariableOp'conv2d2_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0Х
conv2D2/BiasAddBiasAddconv2D2/Conv2D:output:0&conv2D2/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         @j
bn2/ReadVariableOpReadVariableOpbn2_readvariableop_resource*
_output_shapes
:@*
dtype0n
bn2/ReadVariableOp_1ReadVariableOpbn2_readvariableop_1_resource*
_output_shapes
:@*
dtype0М
#bn2/FusedBatchNormV3/ReadVariableOpReadVariableOp,bn2_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype0Р
%bn2/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp.bn2_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype0ъ
bn2/FusedBatchNormV3FusedBatchNormV3conv2D2/BiasAdd:output:0bn2/ReadVariableOp:value:0bn2/ReadVariableOp_1:value:0+bn2/FusedBatchNormV3/ReadVariableOp:value:0-bn2/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:         @:@:@:@:@:*
epsilon%oГ:*
exponential_avg_factor%
╫#<╓
bn2/AssignNewValueAssignVariableOp,bn2_fusedbatchnormv3_readvariableop_resource!bn2/FusedBatchNormV3:batch_mean:0$^bn2/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(р
bn2/AssignNewValue_1AssignVariableOp.bn2_fusedbatchnormv3_readvariableop_1_resource%bn2/FusedBatchNormV3:batch_variance:0&^bn2/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(x
lReLU2/LeakyRelu	LeakyRelubn2/FusedBatchNormV3:y:0*/
_output_shapes
:         @*
alpha%ЪЩЩ>М
conv2D3/Conv2D/ReadVariableOpReadVariableOp&conv2d3_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype0┴
conv2D3/Conv2DConv2DlReLU2/LeakyRelu:activations:0%conv2D3/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         @*
paddingSAME*
strides
В
conv2D3/BiasAdd/ReadVariableOpReadVariableOp'conv2d3_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0Х
conv2D3/BiasAddBiasAddconv2D3/Conv2D:output:0&conv2D3/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         @j
bn3/ReadVariableOpReadVariableOpbn3_readvariableop_resource*
_output_shapes
:@*
dtype0n
bn3/ReadVariableOp_1ReadVariableOpbn3_readvariableop_1_resource*
_output_shapes
:@*
dtype0М
#bn3/FusedBatchNormV3/ReadVariableOpReadVariableOp,bn3_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype0Р
%bn3/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp.bn3_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype0ъ
bn3/FusedBatchNormV3FusedBatchNormV3conv2D3/BiasAdd:output:0bn3/ReadVariableOp:value:0bn3/ReadVariableOp_1:value:0+bn3/FusedBatchNormV3/ReadVariableOp:value:0-bn3/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:         @:@:@:@:@:*
epsilon%oГ:*
exponential_avg_factor%
╫#<╓
bn3/AssignNewValueAssignVariableOp,bn3_fusedbatchnormv3_readvariableop_resource!bn3/FusedBatchNormV3:batch_mean:0$^bn3/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(р
bn3/AssignNewValue_1AssignVariableOp.bn3_fusedbatchnormv3_readvariableop_1_resource%bn3/FusedBatchNormV3:batch_variance:0&^bn3/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(x
lReLU3/LeakyRelu	LeakyRelubn3/FusedBatchNormV3:y:0*/
_output_shapes
:         @*
alpha%ЪЩЩ>М
conv2D4/Conv2D/ReadVariableOpReadVariableOp&conv2d4_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype0┴
conv2D4/Conv2DConv2DlReLU3/LeakyRelu:activations:0%conv2D4/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         @*
paddingSAME*
strides
В
conv2D4/BiasAdd/ReadVariableOpReadVariableOp'conv2d4_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0Х
conv2D4/BiasAddBiasAddconv2D4/Conv2D:output:0&conv2D4/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         @j
bn4/ReadVariableOpReadVariableOpbn4_readvariableop_resource*
_output_shapes
:@*
dtype0n
bn4/ReadVariableOp_1ReadVariableOpbn4_readvariableop_1_resource*
_output_shapes
:@*
dtype0М
#bn4/FusedBatchNormV3/ReadVariableOpReadVariableOp,bn4_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype0Р
%bn4/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp.bn4_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype0ъ
bn4/FusedBatchNormV3FusedBatchNormV3conv2D4/BiasAdd:output:0bn4/ReadVariableOp:value:0bn4/ReadVariableOp_1:value:0+bn4/FusedBatchNormV3/ReadVariableOp:value:0-bn4/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:         @:@:@:@:@:*
epsilon%oГ:*
exponential_avg_factor%
╫#<╓
bn4/AssignNewValueAssignVariableOp,bn4_fusedbatchnormv3_readvariableop_resource!bn4/FusedBatchNormV3:batch_mean:0$^bn4/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(р
bn4/AssignNewValue_1AssignVariableOp.bn4_fusedbatchnormv3_readvariableop_1_resource%bn4/FusedBatchNormV3:batch_variance:0&^bn4/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(x
lReLU4/LeakyRelu	LeakyRelubn4/FusedBatchNormV3:y:0*/
_output_shapes
:         @*
alpha%ЪЩЩ>^
flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"    @  Е
flatten/ReshapeReshapelReLU4/LeakyRelu:activations:0flatten/Const:output:0*
T0*(
_output_shapes
:         └Г
logVar/MatMul/ReadVariableOpReadVariableOp%logvar_matmul_readvariableop_resource*
_output_shapes
:	└
*
dtype0Й
logVar/MatMulMatMulflatten/Reshape:output:0$logVar/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         
А
logVar/BiasAdd/ReadVariableOpReadVariableOp&logvar_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype0Л
logVar/BiasAddBiasAddlogVar/MatMul:product:0%logVar/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         

mean/MatMul/ReadVariableOpReadVariableOp#mean_matmul_readvariableop_resource*
_output_shapes
:	└
*
dtype0Е
mean/MatMulMatMulflatten/Reshape:output:0"mean/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         
|
mean/BiasAdd/ReadVariableOpReadVariableOp$mean_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype0Е
mean/BiasAddBiasAddmean/MatMul:product:0#mean/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         
d
IdentityIdentitymean/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:         
h

Identity_1IdentitylogVar/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:         
▄
NoOpNoOp^bn1/AssignNewValue^bn1/AssignNewValue_1$^bn1/FusedBatchNormV3/ReadVariableOp&^bn1/FusedBatchNormV3/ReadVariableOp_1^bn1/ReadVariableOp^bn1/ReadVariableOp_1^bn2/AssignNewValue^bn2/AssignNewValue_1$^bn2/FusedBatchNormV3/ReadVariableOp&^bn2/FusedBatchNormV3/ReadVariableOp_1^bn2/ReadVariableOp^bn2/ReadVariableOp_1^bn3/AssignNewValue^bn3/AssignNewValue_1$^bn3/FusedBatchNormV3/ReadVariableOp&^bn3/FusedBatchNormV3/ReadVariableOp_1^bn3/ReadVariableOp^bn3/ReadVariableOp_1^bn4/AssignNewValue^bn4/AssignNewValue_1$^bn4/FusedBatchNormV3/ReadVariableOp&^bn4/FusedBatchNormV3/ReadVariableOp_1^bn4/ReadVariableOp^bn4/ReadVariableOp_1^conv2D1/BiasAdd/ReadVariableOp^conv2D1/Conv2D/ReadVariableOp^conv2D2/BiasAdd/ReadVariableOp^conv2D2/Conv2D/ReadVariableOp^conv2D3/BiasAdd/ReadVariableOp^conv2D3/Conv2D/ReadVariableOp^conv2D4/BiasAdd/ReadVariableOp^conv2D4/Conv2D/ReadVariableOp^logVar/BiasAdd/ReadVariableOp^logVar/MatMul/ReadVariableOp^mean/BiasAdd/ReadVariableOp^mean/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "!

identity_1Identity_1:output:0"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*f
_input_shapesU
S:         : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2,
bn1/AssignNewValue_1bn1/AssignNewValue_12(
bn1/AssignNewValuebn1/AssignNewValue2N
%bn1/FusedBatchNormV3/ReadVariableOp_1%bn1/FusedBatchNormV3/ReadVariableOp_12J
#bn1/FusedBatchNormV3/ReadVariableOp#bn1/FusedBatchNormV3/ReadVariableOp2,
bn1/ReadVariableOp_1bn1/ReadVariableOp_12(
bn1/ReadVariableOpbn1/ReadVariableOp2,
bn2/AssignNewValue_1bn2/AssignNewValue_12(
bn2/AssignNewValuebn2/AssignNewValue2N
%bn2/FusedBatchNormV3/ReadVariableOp_1%bn2/FusedBatchNormV3/ReadVariableOp_12J
#bn2/FusedBatchNormV3/ReadVariableOp#bn2/FusedBatchNormV3/ReadVariableOp2,
bn2/ReadVariableOp_1bn2/ReadVariableOp_12(
bn2/ReadVariableOpbn2/ReadVariableOp2,
bn3/AssignNewValue_1bn3/AssignNewValue_12(
bn3/AssignNewValuebn3/AssignNewValue2N
%bn3/FusedBatchNormV3/ReadVariableOp_1%bn3/FusedBatchNormV3/ReadVariableOp_12J
#bn3/FusedBatchNormV3/ReadVariableOp#bn3/FusedBatchNormV3/ReadVariableOp2,
bn3/ReadVariableOp_1bn3/ReadVariableOp_12(
bn3/ReadVariableOpbn3/ReadVariableOp2,
bn4/AssignNewValue_1bn4/AssignNewValue_12(
bn4/AssignNewValuebn4/AssignNewValue2N
%bn4/FusedBatchNormV3/ReadVariableOp_1%bn4/FusedBatchNormV3/ReadVariableOp_12J
#bn4/FusedBatchNormV3/ReadVariableOp#bn4/FusedBatchNormV3/ReadVariableOp2,
bn4/ReadVariableOp_1bn4/ReadVariableOp_12(
bn4/ReadVariableOpbn4/ReadVariableOp2@
conv2D1/BiasAdd/ReadVariableOpconv2D1/BiasAdd/ReadVariableOp2>
conv2D1/Conv2D/ReadVariableOpconv2D1/Conv2D/ReadVariableOp2@
conv2D2/BiasAdd/ReadVariableOpconv2D2/BiasAdd/ReadVariableOp2>
conv2D2/Conv2D/ReadVariableOpconv2D2/Conv2D/ReadVariableOp2@
conv2D3/BiasAdd/ReadVariableOpconv2D3/BiasAdd/ReadVariableOp2>
conv2D3/Conv2D/ReadVariableOpconv2D3/Conv2D/ReadVariableOp2@
conv2D4/BiasAdd/ReadVariableOpconv2D4/BiasAdd/ReadVariableOp2>
conv2D4/Conv2D/ReadVariableOpconv2D4/Conv2D/ReadVariableOp2>
logVar/BiasAdd/ReadVariableOplogVar/BiasAdd/ReadVariableOp2<
logVar/MatMul/ReadVariableOplogVar/MatMul/ReadVariableOp2:
mean/BiasAdd/ReadVariableOpmean/BiasAdd/ReadVariableOp28
mean/MatMul/ReadVariableOpmean/MatMul/ReadVariableOp:W S
/
_output_shapes
:         
 
_user_specified_nameinputs
√
]
A__inference_lReLU4_layer_call_and_return_conditional_losses_10431

inputs
identity_
	LeakyRelu	LeakyReluinputs*/
_output_shapes
:         @*
alpha%ЪЩЩ>g
IdentityIdentityLeakyRelu:activations:0*
T0*/
_output_shapes
:         @"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:         @:W S
/
_output_shapes
:         @
 
_user_specified_nameinputs
е

√
B__inference_conv2D1_layer_call_and_return_conditional_losses_11462

inputs8
conv2d_readvariableop_resource: -
biasadd_readvariableop_resource: 
identityИвBiasAdd/ReadVariableOpвConv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: *
dtype0Щ
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:          *
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
:          g
IdentityIdentityBiasAdd:output:0^NoOp*
T0*/
_output_shapes
:          w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:         : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:         
 
_user_specified_nameinputs
√
]
A__inference_lReLU1_layer_call_and_return_conditional_losses_11534

inputs
identity_
	LeakyRelu	LeakyReluinputs*/
_output_shapes
:          *
alpha%ЪЩЩ>g
IdentityIdentityLeakyRelu:activations:0*
T0*/
_output_shapes
:          "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:          :W S
/
_output_shapes
:          
 
_user_specified_nameinputs
ы
╛
#__inference_bn1_layer_call_fn_11488

inputs
unknown: 
	unknown_0: 
	unknown_1: 
	unknown_2: 
identityИвStatefulPartitionedCallЗ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+                            *&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *G
fBR@
>__inference_bn1_layer_call_and_return_conditional_losses_10082Й
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+                            `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+                            : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+                            
 
_user_specified_nameinputs
╙y
В
 __inference__wrapped_model_10045
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
6encoder_bn4_fusedbatchnormv3_readvariableop_1_resource:@@
-encoder_logvar_matmul_readvariableop_resource:	└
<
.encoder_logvar_biasadd_readvariableop_resource:
>
+encoder_mean_matmul_readvariableop_resource:	└
:
,encoder_mean_biasadd_readvariableop_resource:

identity

identity_1Ив+encoder/bn1/FusedBatchNormV3/ReadVariableOpв-encoder/bn1/FusedBatchNormV3/ReadVariableOp_1вencoder/bn1/ReadVariableOpвencoder/bn1/ReadVariableOp_1в+encoder/bn2/FusedBatchNormV3/ReadVariableOpв-encoder/bn2/FusedBatchNormV3/ReadVariableOp_1вencoder/bn2/ReadVariableOpвencoder/bn2/ReadVariableOp_1в+encoder/bn3/FusedBatchNormV3/ReadVariableOpв-encoder/bn3/FusedBatchNormV3/ReadVariableOp_1вencoder/bn3/ReadVariableOpвencoder/bn3/ReadVariableOp_1в+encoder/bn4/FusedBatchNormV3/ReadVariableOpв-encoder/bn4/FusedBatchNormV3/ReadVariableOp_1вencoder/bn4/ReadVariableOpвencoder/bn4/ReadVariableOp_1в&encoder/conv2D1/BiasAdd/ReadVariableOpв%encoder/conv2D1/Conv2D/ReadVariableOpв&encoder/conv2D2/BiasAdd/ReadVariableOpв%encoder/conv2D2/Conv2D/ReadVariableOpв&encoder/conv2D3/BiasAdd/ReadVariableOpв%encoder/conv2D3/Conv2D/ReadVariableOpв&encoder/conv2D4/BiasAdd/ReadVariableOpв%encoder/conv2D4/Conv2D/ReadVariableOpв%encoder/logVar/BiasAdd/ReadVariableOpв$encoder/logVar/MatMul/ReadVariableOpв#encoder/mean/BiasAdd/ReadVariableOpв"encoder/mean/MatMul/ReadVariableOpЬ
%encoder/conv2D1/Conv2D/ReadVariableOpReadVariableOp.encoder_conv2d1_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype0╛
encoder/conv2D1/Conv2DConv2Dinput_layer-encoder/conv2D1/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:          *
paddingSAME*
strides
Т
&encoder/conv2D1/BiasAdd/ReadVariableOpReadVariableOp/encoder_conv2d1_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0н
encoder/conv2D1/BiasAddBiasAddencoder/conv2D1/Conv2D:output:0.encoder/conv2D1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:          z
encoder/bn1/ReadVariableOpReadVariableOp#encoder_bn1_readvariableop_resource*
_output_shapes
: *
dtype0~
encoder/bn1/ReadVariableOp_1ReadVariableOp%encoder_bn1_readvariableop_1_resource*
_output_shapes
: *
dtype0Ь
+encoder/bn1/FusedBatchNormV3/ReadVariableOpReadVariableOp4encoder_bn1_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype0а
-encoder/bn1/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp6encoder_bn1_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype0М
encoder/bn1/FusedBatchNormV3FusedBatchNormV3 encoder/conv2D1/BiasAdd:output:0"encoder/bn1/ReadVariableOp:value:0$encoder/bn1/ReadVariableOp_1:value:03encoder/bn1/FusedBatchNormV3/ReadVariableOp:value:05encoder/bn1/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:          : : : : :*
epsilon%oГ:*
is_training( И
encoder/lReLU1/LeakyRelu	LeakyRelu encoder/bn1/FusedBatchNormV3:y:0*/
_output_shapes
:          *
alpha%ЪЩЩ>Ь
%encoder/conv2D2/Conv2D/ReadVariableOpReadVariableOp.encoder_conv2d2_conv2d_readvariableop_resource*&
_output_shapes
: @*
dtype0┘
encoder/conv2D2/Conv2DConv2D&encoder/lReLU1/LeakyRelu:activations:0-encoder/conv2D2/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         @*
paddingSAME*
strides
Т
&encoder/conv2D2/BiasAdd/ReadVariableOpReadVariableOp/encoder_conv2d2_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0н
encoder/conv2D2/BiasAddBiasAddencoder/conv2D2/Conv2D:output:0.encoder/conv2D2/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         @z
encoder/bn2/ReadVariableOpReadVariableOp#encoder_bn2_readvariableop_resource*
_output_shapes
:@*
dtype0~
encoder/bn2/ReadVariableOp_1ReadVariableOp%encoder_bn2_readvariableop_1_resource*
_output_shapes
:@*
dtype0Ь
+encoder/bn2/FusedBatchNormV3/ReadVariableOpReadVariableOp4encoder_bn2_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype0а
-encoder/bn2/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp6encoder_bn2_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype0М
encoder/bn2/FusedBatchNormV3FusedBatchNormV3 encoder/conv2D2/BiasAdd:output:0"encoder/bn2/ReadVariableOp:value:0$encoder/bn2/ReadVariableOp_1:value:03encoder/bn2/FusedBatchNormV3/ReadVariableOp:value:05encoder/bn2/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:         @:@:@:@:@:*
epsilon%oГ:*
is_training( И
encoder/lReLU2/LeakyRelu	LeakyRelu encoder/bn2/FusedBatchNormV3:y:0*/
_output_shapes
:         @*
alpha%ЪЩЩ>Ь
%encoder/conv2D3/Conv2D/ReadVariableOpReadVariableOp.encoder_conv2d3_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype0┘
encoder/conv2D3/Conv2DConv2D&encoder/lReLU2/LeakyRelu:activations:0-encoder/conv2D3/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         @*
paddingSAME*
strides
Т
&encoder/conv2D3/BiasAdd/ReadVariableOpReadVariableOp/encoder_conv2d3_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0н
encoder/conv2D3/BiasAddBiasAddencoder/conv2D3/Conv2D:output:0.encoder/conv2D3/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         @z
encoder/bn3/ReadVariableOpReadVariableOp#encoder_bn3_readvariableop_resource*
_output_shapes
:@*
dtype0~
encoder/bn3/ReadVariableOp_1ReadVariableOp%encoder_bn3_readvariableop_1_resource*
_output_shapes
:@*
dtype0Ь
+encoder/bn3/FusedBatchNormV3/ReadVariableOpReadVariableOp4encoder_bn3_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype0а
-encoder/bn3/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp6encoder_bn3_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype0М
encoder/bn3/FusedBatchNormV3FusedBatchNormV3 encoder/conv2D3/BiasAdd:output:0"encoder/bn3/ReadVariableOp:value:0$encoder/bn3/ReadVariableOp_1:value:03encoder/bn3/FusedBatchNormV3/ReadVariableOp:value:05encoder/bn3/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:         @:@:@:@:@:*
epsilon%oГ:*
is_training( И
encoder/lReLU3/LeakyRelu	LeakyRelu encoder/bn3/FusedBatchNormV3:y:0*/
_output_shapes
:         @*
alpha%ЪЩЩ>Ь
%encoder/conv2D4/Conv2D/ReadVariableOpReadVariableOp.encoder_conv2d4_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype0┘
encoder/conv2D4/Conv2DConv2D&encoder/lReLU3/LeakyRelu:activations:0-encoder/conv2D4/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         @*
paddingSAME*
strides
Т
&encoder/conv2D4/BiasAdd/ReadVariableOpReadVariableOp/encoder_conv2d4_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0н
encoder/conv2D4/BiasAddBiasAddencoder/conv2D4/Conv2D:output:0.encoder/conv2D4/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         @z
encoder/bn4/ReadVariableOpReadVariableOp#encoder_bn4_readvariableop_resource*
_output_shapes
:@*
dtype0~
encoder/bn4/ReadVariableOp_1ReadVariableOp%encoder_bn4_readvariableop_1_resource*
_output_shapes
:@*
dtype0Ь
+encoder/bn4/FusedBatchNormV3/ReadVariableOpReadVariableOp4encoder_bn4_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype0а
-encoder/bn4/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp6encoder_bn4_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype0М
encoder/bn4/FusedBatchNormV3FusedBatchNormV3 encoder/conv2D4/BiasAdd:output:0"encoder/bn4/ReadVariableOp:value:0$encoder/bn4/ReadVariableOp_1:value:03encoder/bn4/FusedBatchNormV3/ReadVariableOp:value:05encoder/bn4/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:         @:@:@:@:@:*
epsilon%oГ:*
is_training( И
encoder/lReLU4/LeakyRelu	LeakyRelu encoder/bn4/FusedBatchNormV3:y:0*/
_output_shapes
:         @*
alpha%ЪЩЩ>f
encoder/flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"    @  Э
encoder/flatten/ReshapeReshape&encoder/lReLU4/LeakyRelu:activations:0encoder/flatten/Const:output:0*
T0*(
_output_shapes
:         └У
$encoder/logVar/MatMul/ReadVariableOpReadVariableOp-encoder_logvar_matmul_readvariableop_resource*
_output_shapes
:	└
*
dtype0б
encoder/logVar/MatMulMatMul encoder/flatten/Reshape:output:0,encoder/logVar/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         
Р
%encoder/logVar/BiasAdd/ReadVariableOpReadVariableOp.encoder_logvar_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype0г
encoder/logVar/BiasAddBiasAddencoder/logVar/MatMul:product:0-encoder/logVar/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         
П
"encoder/mean/MatMul/ReadVariableOpReadVariableOp+encoder_mean_matmul_readvariableop_resource*
_output_shapes
:	└
*
dtype0Э
encoder/mean/MatMulMatMul encoder/flatten/Reshape:output:0*encoder/mean/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         
М
#encoder/mean/BiasAdd/ReadVariableOpReadVariableOp,encoder_mean_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype0Э
encoder/mean/BiasAddBiasAddencoder/mean/MatMul:product:0+encoder/mean/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         
n
IdentityIdentityencoder/logVar/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:         
n

Identity_1Identityencoder/mean/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:         
М	
NoOpNoOp,^encoder/bn1/FusedBatchNormV3/ReadVariableOp.^encoder/bn1/FusedBatchNormV3/ReadVariableOp_1^encoder/bn1/ReadVariableOp^encoder/bn1/ReadVariableOp_1,^encoder/bn2/FusedBatchNormV3/ReadVariableOp.^encoder/bn2/FusedBatchNormV3/ReadVariableOp_1^encoder/bn2/ReadVariableOp^encoder/bn2/ReadVariableOp_1,^encoder/bn3/FusedBatchNormV3/ReadVariableOp.^encoder/bn3/FusedBatchNormV3/ReadVariableOp_1^encoder/bn3/ReadVariableOp^encoder/bn3/ReadVariableOp_1,^encoder/bn4/FusedBatchNormV3/ReadVariableOp.^encoder/bn4/FusedBatchNormV3/ReadVariableOp_1^encoder/bn4/ReadVariableOp^encoder/bn4/ReadVariableOp_1'^encoder/conv2D1/BiasAdd/ReadVariableOp&^encoder/conv2D1/Conv2D/ReadVariableOp'^encoder/conv2D2/BiasAdd/ReadVariableOp&^encoder/conv2D2/Conv2D/ReadVariableOp'^encoder/conv2D3/BiasAdd/ReadVariableOp&^encoder/conv2D3/Conv2D/ReadVariableOp'^encoder/conv2D4/BiasAdd/ReadVariableOp&^encoder/conv2D4/Conv2D/ReadVariableOp&^encoder/logVar/BiasAdd/ReadVariableOp%^encoder/logVar/MatMul/ReadVariableOp$^encoder/mean/BiasAdd/ReadVariableOp#^encoder/mean/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "!

identity_1Identity_1:output:0"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*f
_input_shapesU
S:         : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2^
-encoder/bn1/FusedBatchNormV3/ReadVariableOp_1-encoder/bn1/FusedBatchNormV3/ReadVariableOp_12Z
+encoder/bn1/FusedBatchNormV3/ReadVariableOp+encoder/bn1/FusedBatchNormV3/ReadVariableOp2<
encoder/bn1/ReadVariableOp_1encoder/bn1/ReadVariableOp_128
encoder/bn1/ReadVariableOpencoder/bn1/ReadVariableOp2^
-encoder/bn2/FusedBatchNormV3/ReadVariableOp_1-encoder/bn2/FusedBatchNormV3/ReadVariableOp_12Z
+encoder/bn2/FusedBatchNormV3/ReadVariableOp+encoder/bn2/FusedBatchNormV3/ReadVariableOp2<
encoder/bn2/ReadVariableOp_1encoder/bn2/ReadVariableOp_128
encoder/bn2/ReadVariableOpencoder/bn2/ReadVariableOp2^
-encoder/bn3/FusedBatchNormV3/ReadVariableOp_1-encoder/bn3/FusedBatchNormV3/ReadVariableOp_12Z
+encoder/bn3/FusedBatchNormV3/ReadVariableOp+encoder/bn3/FusedBatchNormV3/ReadVariableOp2<
encoder/bn3/ReadVariableOp_1encoder/bn3/ReadVariableOp_128
encoder/bn3/ReadVariableOpencoder/bn3/ReadVariableOp2^
-encoder/bn4/FusedBatchNormV3/ReadVariableOp_1-encoder/bn4/FusedBatchNormV3/ReadVariableOp_12Z
+encoder/bn4/FusedBatchNormV3/ReadVariableOp+encoder/bn4/FusedBatchNormV3/ReadVariableOp2<
encoder/bn4/ReadVariableOp_1encoder/bn4/ReadVariableOp_128
encoder/bn4/ReadVariableOpencoder/bn4/ReadVariableOp2P
&encoder/conv2D1/BiasAdd/ReadVariableOp&encoder/conv2D1/BiasAdd/ReadVariableOp2N
%encoder/conv2D1/Conv2D/ReadVariableOp%encoder/conv2D1/Conv2D/ReadVariableOp2P
&encoder/conv2D2/BiasAdd/ReadVariableOp&encoder/conv2D2/BiasAdd/ReadVariableOp2N
%encoder/conv2D2/Conv2D/ReadVariableOp%encoder/conv2D2/Conv2D/ReadVariableOp2P
&encoder/conv2D3/BiasAdd/ReadVariableOp&encoder/conv2D3/BiasAdd/ReadVariableOp2N
%encoder/conv2D3/Conv2D/ReadVariableOp%encoder/conv2D3/Conv2D/ReadVariableOp2P
&encoder/conv2D4/BiasAdd/ReadVariableOp&encoder/conv2D4/BiasAdd/ReadVariableOp2N
%encoder/conv2D4/Conv2D/ReadVariableOp%encoder/conv2D4/Conv2D/ReadVariableOp2N
%encoder/logVar/BiasAdd/ReadVariableOp%encoder/logVar/BiasAdd/ReadVariableOp2L
$encoder/logVar/MatMul/ReadVariableOp$encoder/logVar/MatMul/ReadVariableOp2J
#encoder/mean/BiasAdd/ReadVariableOp#encoder/mean/BiasAdd/ReadVariableOp2H
"encoder/mean/MatMul/ReadVariableOp"encoder/mean/MatMul/ReadVariableOp:\ X
/
_output_shapes
:         
%
_user_specified_nameinput_layer
щ
╛
#__inference_bn4_layer_call_fn_11748

inputs
unknown:@
	unknown_0:@
	unknown_1:@
	unknown_2:@
identityИвStatefulPartitionedCallЕ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+                           @*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *G
fBR@
>__inference_bn4_layer_call_and_return_conditional_losses_10256Й
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+                           @`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+                           @: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+                           @
 
_user_specified_nameinputs
ы
╛
#__inference_bn2_layer_call_fn_11579

inputs
unknown:@
	unknown_0:@
	unknown_1:@
	unknown_2:@
identityИвStatefulPartitionedCallЗ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+                           @*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *G
fBR@
>__inference_bn2_layer_call_and_return_conditional_losses_10146Й
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+                           @`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+                           @: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+                           @
 
_user_specified_nameinputs
Ё
ї
'__inference_encoder_layer_call_fn_11237

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

unknown_22:@

unknown_23:	└


unknown_24:


unknown_25:	└


unknown_26:

identity

identity_1ИвStatefulPartitionedCall╬
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
unknown_26*(
Tin!
2*
Tout
2*
_collective_manager_ids
 *:
_output_shapes(
&:         
:         
*>
_read_only_resource_inputs 
	
*-
config_proto

CPU

GPU 2J 8В *K
fFRD
B__inference_encoder_layer_call_and_return_conditional_losses_10769o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         
q

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*'
_output_shapes
:         
`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "!

identity_1Identity_1:output:0"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*f
_input_shapesU
S:         : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:         
 
_user_specified_nameinputs
е

√
B__inference_conv2D2_layer_call_and_return_conditional_losses_10347

inputs8
conv2d_readvariableop_resource: @-
biasadd_readvariableop_resource:@
identityИвBiasAdd/ReadVariableOpвConv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: @*
dtype0Щ
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         @*
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
:         @g
IdentityIdentityBiasAdd:output:0^NoOp*
T0*/
_output_shapes
:         @w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:          : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:          
 
_user_specified_nameinputs
√
]
A__inference_lReLU4_layer_call_and_return_conditional_losses_11807

inputs
identity_
	LeakyRelu	LeakyReluinputs*/
_output_shapes
:         @*
alpha%ЪЩЩ>g
IdentityIdentityLeakyRelu:activations:0*
T0*/
_output_shapes
:         @"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:         @:W S
/
_output_shapes
:         @
 
_user_specified_nameinputs
╣
Й
>__inference_bn2_layer_call_and_return_conditional_losses_10146

inputs%
readvariableop_resource:@'
readvariableop_1_resource:@6
(fusedbatchnormv3_readvariableop_resource:@8
*fusedbatchnormv3_readvariableop_1_resource:@
identityИвFusedBatchNormV3/ReadVariableOpв!FusedBatchNormV3/ReadVariableOp_1вReadVariableOpвReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype0Д
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype0И
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype0╚
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+                           @:@:@:@:@:*
epsilon%oГ:*
is_training( }
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+                           @░
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+                           @: : : : 2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2$
ReadVariableOp_1ReadVariableOp_12 
ReadVariableOpReadVariableOp:i e
A
_output_shapes/
-:+                           @
 
_user_specified_nameinputs
є
н
>__inference_bn1_layer_call_and_return_conditional_losses_11506

inputs%
readvariableop_resource: '
readvariableop_1_resource: 6
(fusedbatchnormv3_readvariableop_resource: 8
*fusedbatchnormv3_readvariableop_1_resource: 
identityИвAssignNewValueвAssignNewValue_1вFusedBatchNormV3/ReadVariableOpв!FusedBatchNormV3/ReadVariableOp_1вReadVariableOpвReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype0Д
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype0И
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype0╓
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+                            : : : : :*
epsilon%oГ:*
exponential_avg_factor%
╫#<╞
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(╨
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(}
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+                            ╘
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+                            : : : : 2$
AssignNewValue_1AssignNewValue_12 
AssignNewValueAssignNewValue2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2$
ReadVariableOp_1ReadVariableOp_12 
ReadVariableOpReadVariableOp:i e
A
_output_shapes/
-:+                            
 
_user_specified_nameinputs
╣
Й
>__inference_bn2_layer_call_and_return_conditional_losses_11615

inputs%
readvariableop_resource:@'
readvariableop_1_resource:@6
(fusedbatchnormv3_readvariableop_resource:@8
*fusedbatchnormv3_readvariableop_1_resource:@
identityИвFusedBatchNormV3/ReadVariableOpв!FusedBatchNormV3/ReadVariableOp_1вReadVariableOpвReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype0Д
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype0И
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype0╚
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+                           @:@:@:@:@:*
epsilon%oГ:*
is_training( }
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+                           @░
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+                           @: : : : 2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2$
ReadVariableOp_1ReadVariableOp_12 
ReadVariableOpReadVariableOp:i e
A
_output_shapes/
-:+                           @
 
_user_specified_nameinputs
╖
B
&__inference_lReLU1_layer_call_fn_11529

inputs
identity┤
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:          * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *J
fERC
A__inference_lReLU1_layer_call_and_return_conditional_losses_10335h
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:          "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:          :W S
/
_output_shapes
:          
 
_user_specified_nameinputs
ы
╛
#__inference_bn4_layer_call_fn_11761

inputs
unknown:@
	unknown_0:@
	unknown_1:@
	unknown_2:@
identityИвStatefulPartitionedCallЗ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+                           @*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *G
fBR@
>__inference_bn4_layer_call_and_return_conditional_losses_10274Й
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+                           @`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+                           @: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+                           @
 
_user_specified_nameinputs
√
]
A__inference_lReLU3_layer_call_and_return_conditional_losses_10399

inputs
identity_
	LeakyRelu	LeakyReluinputs*/
_output_shapes
:         @*
alpha%ЪЩЩ>g
IdentityIdentityLeakyRelu:activations:0*
T0*/
_output_shapes
:         @"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:         @:W S
/
_output_shapes
:         @
 
_user_specified_nameinputs
┐
Ф
&__inference_logVar_layer_call_fn_11846

inputs
unknown:	└

	unknown_0:

identityИвStatefulPartitionedCall╓
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         
*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *J
fERC
A__inference_logVar_layer_call_and_return_conditional_losses_10451o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         
`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:         └: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:         └
 
_user_specified_nameinputs
╖
B
&__inference_lReLU2_layer_call_fn_11620

inputs
identity┤
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         @* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *J
fERC
A__inference_lReLU2_layer_call_and_return_conditional_losses_10367h
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:         @"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:         @:W S
/
_output_shapes
:         @
 
_user_specified_nameinputs
ў
·
'__inference_encoder_layer_call_fn_10691
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

unknown_22:@

unknown_23:	└


unknown_24:


unknown_25:	└


unknown_26:

identity

identity_1ИвStatefulPartitionedCall╦
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
unknown_26*(
Tin!
2*
Tout
2*
_collective_manager_ids
 *:
_output_shapes(
&:         
:         
*6
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8В *K
fFRD
B__inference_encoder_layer_call_and_return_conditional_losses_10630o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         
q

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*'
_output_shapes
:         
`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "!

identity_1Identity_1:output:0"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*f
_input_shapesU
S:         : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:\ X
/
_output_shapes
:         
%
_user_specified_nameinput_layer
°h
▀
B__inference_encoder_layer_call_and_return_conditional_losses_11443

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
.bn4_fusedbatchnormv3_readvariableop_1_resource:@8
%logvar_matmul_readvariableop_resource:	└
4
&logvar_biasadd_readvariableop_resource:
6
#mean_matmul_readvariableop_resource:	└
2
$mean_biasadd_readvariableop_resource:

identity

identity_1Ив#bn1/FusedBatchNormV3/ReadVariableOpв%bn1/FusedBatchNormV3/ReadVariableOp_1вbn1/ReadVariableOpвbn1/ReadVariableOp_1в#bn2/FusedBatchNormV3/ReadVariableOpв%bn2/FusedBatchNormV3/ReadVariableOp_1вbn2/ReadVariableOpвbn2/ReadVariableOp_1в#bn3/FusedBatchNormV3/ReadVariableOpв%bn3/FusedBatchNormV3/ReadVariableOp_1вbn3/ReadVariableOpвbn3/ReadVariableOp_1в#bn4/FusedBatchNormV3/ReadVariableOpв%bn4/FusedBatchNormV3/ReadVariableOp_1вbn4/ReadVariableOpвbn4/ReadVariableOp_1вconv2D1/BiasAdd/ReadVariableOpвconv2D1/Conv2D/ReadVariableOpвconv2D2/BiasAdd/ReadVariableOpвconv2D2/Conv2D/ReadVariableOpвconv2D3/BiasAdd/ReadVariableOpвconv2D3/Conv2D/ReadVariableOpвconv2D4/BiasAdd/ReadVariableOpвconv2D4/Conv2D/ReadVariableOpвlogVar/BiasAdd/ReadVariableOpвlogVar/MatMul/ReadVariableOpвmean/BiasAdd/ReadVariableOpвmean/MatMul/ReadVariableOpМ
conv2D1/Conv2D/ReadVariableOpReadVariableOp&conv2d1_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype0й
conv2D1/Conv2DConv2Dinputs%conv2D1/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:          *
paddingSAME*
strides
В
conv2D1/BiasAdd/ReadVariableOpReadVariableOp'conv2d1_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0Х
conv2D1/BiasAddBiasAddconv2D1/Conv2D:output:0&conv2D1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:          j
bn1/ReadVariableOpReadVariableOpbn1_readvariableop_resource*
_output_shapes
: *
dtype0n
bn1/ReadVariableOp_1ReadVariableOpbn1_readvariableop_1_resource*
_output_shapes
: *
dtype0М
#bn1/FusedBatchNormV3/ReadVariableOpReadVariableOp,bn1_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype0Р
%bn1/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp.bn1_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype0▄
bn1/FusedBatchNormV3FusedBatchNormV3conv2D1/BiasAdd:output:0bn1/ReadVariableOp:value:0bn1/ReadVariableOp_1:value:0+bn1/FusedBatchNormV3/ReadVariableOp:value:0-bn1/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:          : : : : :*
epsilon%oГ:*
is_training( x
lReLU1/LeakyRelu	LeakyRelubn1/FusedBatchNormV3:y:0*/
_output_shapes
:          *
alpha%ЪЩЩ>М
conv2D2/Conv2D/ReadVariableOpReadVariableOp&conv2d2_conv2d_readvariableop_resource*&
_output_shapes
: @*
dtype0┴
conv2D2/Conv2DConv2DlReLU1/LeakyRelu:activations:0%conv2D2/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         @*
paddingSAME*
strides
В
conv2D2/BiasAdd/ReadVariableOpReadVariableOp'conv2d2_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0Х
conv2D2/BiasAddBiasAddconv2D2/Conv2D:output:0&conv2D2/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         @j
bn2/ReadVariableOpReadVariableOpbn2_readvariableop_resource*
_output_shapes
:@*
dtype0n
bn2/ReadVariableOp_1ReadVariableOpbn2_readvariableop_1_resource*
_output_shapes
:@*
dtype0М
#bn2/FusedBatchNormV3/ReadVariableOpReadVariableOp,bn2_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype0Р
%bn2/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp.bn2_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype0▄
bn2/FusedBatchNormV3FusedBatchNormV3conv2D2/BiasAdd:output:0bn2/ReadVariableOp:value:0bn2/ReadVariableOp_1:value:0+bn2/FusedBatchNormV3/ReadVariableOp:value:0-bn2/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:         @:@:@:@:@:*
epsilon%oГ:*
is_training( x
lReLU2/LeakyRelu	LeakyRelubn2/FusedBatchNormV3:y:0*/
_output_shapes
:         @*
alpha%ЪЩЩ>М
conv2D3/Conv2D/ReadVariableOpReadVariableOp&conv2d3_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype0┴
conv2D3/Conv2DConv2DlReLU2/LeakyRelu:activations:0%conv2D3/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         @*
paddingSAME*
strides
В
conv2D3/BiasAdd/ReadVariableOpReadVariableOp'conv2d3_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0Х
conv2D3/BiasAddBiasAddconv2D3/Conv2D:output:0&conv2D3/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         @j
bn3/ReadVariableOpReadVariableOpbn3_readvariableop_resource*
_output_shapes
:@*
dtype0n
bn3/ReadVariableOp_1ReadVariableOpbn3_readvariableop_1_resource*
_output_shapes
:@*
dtype0М
#bn3/FusedBatchNormV3/ReadVariableOpReadVariableOp,bn3_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype0Р
%bn3/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp.bn3_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype0▄
bn3/FusedBatchNormV3FusedBatchNormV3conv2D3/BiasAdd:output:0bn3/ReadVariableOp:value:0bn3/ReadVariableOp_1:value:0+bn3/FusedBatchNormV3/ReadVariableOp:value:0-bn3/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:         @:@:@:@:@:*
epsilon%oГ:*
is_training( x
lReLU3/LeakyRelu	LeakyRelubn3/FusedBatchNormV3:y:0*/
_output_shapes
:         @*
alpha%ЪЩЩ>М
conv2D4/Conv2D/ReadVariableOpReadVariableOp&conv2d4_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype0┴
conv2D4/Conv2DConv2DlReLU3/LeakyRelu:activations:0%conv2D4/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         @*
paddingSAME*
strides
В
conv2D4/BiasAdd/ReadVariableOpReadVariableOp'conv2d4_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0Х
conv2D4/BiasAddBiasAddconv2D4/Conv2D:output:0&conv2D4/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         @j
bn4/ReadVariableOpReadVariableOpbn4_readvariableop_resource*
_output_shapes
:@*
dtype0n
bn4/ReadVariableOp_1ReadVariableOpbn4_readvariableop_1_resource*
_output_shapes
:@*
dtype0М
#bn4/FusedBatchNormV3/ReadVariableOpReadVariableOp,bn4_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype0Р
%bn4/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp.bn4_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype0▄
bn4/FusedBatchNormV3FusedBatchNormV3conv2D4/BiasAdd:output:0bn4/ReadVariableOp:value:0bn4/ReadVariableOp_1:value:0+bn4/FusedBatchNormV3/ReadVariableOp:value:0-bn4/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:         @:@:@:@:@:*
epsilon%oГ:*
is_training( x
lReLU4/LeakyRelu	LeakyRelubn4/FusedBatchNormV3:y:0*/
_output_shapes
:         @*
alpha%ЪЩЩ>^
flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"    @  Е
flatten/ReshapeReshapelReLU4/LeakyRelu:activations:0flatten/Const:output:0*
T0*(
_output_shapes
:         └Г
logVar/MatMul/ReadVariableOpReadVariableOp%logvar_matmul_readvariableop_resource*
_output_shapes
:	└
*
dtype0Й
logVar/MatMulMatMulflatten/Reshape:output:0$logVar/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         
А
logVar/BiasAdd/ReadVariableOpReadVariableOp&logvar_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype0Л
logVar/BiasAddBiasAddlogVar/MatMul:product:0%logVar/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         

mean/MatMul/ReadVariableOpReadVariableOp#mean_matmul_readvariableop_resource*
_output_shapes
:	└
*
dtype0Е
mean/MatMulMatMulflatten/Reshape:output:0"mean/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         
|
mean/BiasAdd/ReadVariableOpReadVariableOp$mean_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype0Е
mean/BiasAddBiasAddmean/MatMul:product:0#mean/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         
d
IdentityIdentitymean/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:         
h

Identity_1IdentitylogVar/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:         
м
NoOpNoOp$^bn1/FusedBatchNormV3/ReadVariableOp&^bn1/FusedBatchNormV3/ReadVariableOp_1^bn1/ReadVariableOp^bn1/ReadVariableOp_1$^bn2/FusedBatchNormV3/ReadVariableOp&^bn2/FusedBatchNormV3/ReadVariableOp_1^bn2/ReadVariableOp^bn2/ReadVariableOp_1$^bn3/FusedBatchNormV3/ReadVariableOp&^bn3/FusedBatchNormV3/ReadVariableOp_1^bn3/ReadVariableOp^bn3/ReadVariableOp_1$^bn4/FusedBatchNormV3/ReadVariableOp&^bn4/FusedBatchNormV3/ReadVariableOp_1^bn4/ReadVariableOp^bn4/ReadVariableOp_1^conv2D1/BiasAdd/ReadVariableOp^conv2D1/Conv2D/ReadVariableOp^conv2D2/BiasAdd/ReadVariableOp^conv2D2/Conv2D/ReadVariableOp^conv2D3/BiasAdd/ReadVariableOp^conv2D3/Conv2D/ReadVariableOp^conv2D4/BiasAdd/ReadVariableOp^conv2D4/Conv2D/ReadVariableOp^logVar/BiasAdd/ReadVariableOp^logVar/MatMul/ReadVariableOp^mean/BiasAdd/ReadVariableOp^mean/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "!

identity_1Identity_1:output:0"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*f
_input_shapesU
S:         : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2N
%bn1/FusedBatchNormV3/ReadVariableOp_1%bn1/FusedBatchNormV3/ReadVariableOp_12J
#bn1/FusedBatchNormV3/ReadVariableOp#bn1/FusedBatchNormV3/ReadVariableOp2,
bn1/ReadVariableOp_1bn1/ReadVariableOp_12(
bn1/ReadVariableOpbn1/ReadVariableOp2N
%bn2/FusedBatchNormV3/ReadVariableOp_1%bn2/FusedBatchNormV3/ReadVariableOp_12J
#bn2/FusedBatchNormV3/ReadVariableOp#bn2/FusedBatchNormV3/ReadVariableOp2,
bn2/ReadVariableOp_1bn2/ReadVariableOp_12(
bn2/ReadVariableOpbn2/ReadVariableOp2N
%bn3/FusedBatchNormV3/ReadVariableOp_1%bn3/FusedBatchNormV3/ReadVariableOp_12J
#bn3/FusedBatchNormV3/ReadVariableOp#bn3/FusedBatchNormV3/ReadVariableOp2,
bn3/ReadVariableOp_1bn3/ReadVariableOp_12(
bn3/ReadVariableOpbn3/ReadVariableOp2N
%bn4/FusedBatchNormV3/ReadVariableOp_1%bn4/FusedBatchNormV3/ReadVariableOp_12J
#bn4/FusedBatchNormV3/ReadVariableOp#bn4/FusedBatchNormV3/ReadVariableOp2,
bn4/ReadVariableOp_1bn4/ReadVariableOp_12(
bn4/ReadVariableOpbn4/ReadVariableOp2@
conv2D1/BiasAdd/ReadVariableOpconv2D1/BiasAdd/ReadVariableOp2>
conv2D1/Conv2D/ReadVariableOpconv2D1/Conv2D/ReadVariableOp2@
conv2D2/BiasAdd/ReadVariableOpconv2D2/BiasAdd/ReadVariableOp2>
conv2D2/Conv2D/ReadVariableOpconv2D2/Conv2D/ReadVariableOp2@
conv2D3/BiasAdd/ReadVariableOpconv2D3/BiasAdd/ReadVariableOp2>
conv2D3/Conv2D/ReadVariableOpconv2D3/Conv2D/ReadVariableOp2@
conv2D4/BiasAdd/ReadVariableOpconv2D4/BiasAdd/ReadVariableOp2>
conv2D4/Conv2D/ReadVariableOpconv2D4/Conv2D/ReadVariableOp2>
logVar/BiasAdd/ReadVariableOplogVar/BiasAdd/ReadVariableOp2<
logVar/MatMul/ReadVariableOplogVar/MatMul/ReadVariableOp2:
mean/BiasAdd/ReadVariableOpmean/BiasAdd/ReadVariableOp28
mean/MatMul/ReadVariableOpmean/MatMul/ReadVariableOp:W S
/
_output_shapes
:         
 
_user_specified_nameinputs
є
н
>__inference_bn4_layer_call_and_return_conditional_losses_10256

inputs%
readvariableop_resource:@'
readvariableop_1_resource:@6
(fusedbatchnormv3_readvariableop_resource:@8
*fusedbatchnormv3_readvariableop_1_resource:@
identityИвAssignNewValueвAssignNewValue_1вFusedBatchNormV3/ReadVariableOpв!FusedBatchNormV3/ReadVariableOp_1вReadVariableOpвReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype0Д
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype0И
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype0╓
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+                           @:@:@:@:@:*
epsilon%oГ:*
exponential_avg_factor%
╫#<╞
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(╨
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(}
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+                           @╘
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+                           @: : : : 2$
AssignNewValue_1AssignNewValue_12 
AssignNewValueAssignNewValue2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2$
ReadVariableOp_1ReadVariableOp_12 
ReadVariableOpReadVariableOp:i e
A
_output_shapes/
-:+                           @
 
_user_specified_nameinputs
╖
B
&__inference_lReLU3_layer_call_fn_11711

inputs
identity┤
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         @* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *J
fERC
A__inference_lReLU3_layer_call_and_return_conditional_losses_10399h
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:         @"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:         @:W S
/
_output_shapes
:         @
 
_user_specified_nameinputs
є
н
>__inference_bn4_layer_call_and_return_conditional_losses_11779

inputs%
readvariableop_resource:@'
readvariableop_1_resource:@6
(fusedbatchnormv3_readvariableop_resource:@8
*fusedbatchnormv3_readvariableop_1_resource:@
identityИвAssignNewValueвAssignNewValue_1вFusedBatchNormV3/ReadVariableOpв!FusedBatchNormV3/ReadVariableOp_1вReadVariableOpвReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype0Д
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype0И
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype0╓
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+                           @:@:@:@:@:*
epsilon%oГ:*
exponential_avg_factor%
╫#<╞
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(╨
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(}
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+                           @╘
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+                           @: : : : 2$
AssignNewValue_1AssignNewValue_12 
AssignNewValueAssignNewValue2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2$
ReadVariableOp_1ReadVariableOp_12 
ReadVariableOpReadVariableOp:i e
A
_output_shapes/
-:+                           @
 
_user_specified_nameinputs
е

√
B__inference_conv2D1_layer_call_and_return_conditional_losses_10315

inputs8
conv2d_readvariableop_resource: -
biasadd_readvariableop_resource: 
identityИвBiasAdd/ReadVariableOpвConv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: *
dtype0Щ
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:          *
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
:          g
IdentityIdentityBiasAdd:output:0^NoOp*
T0*/
_output_shapes
:          w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:         : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:         
 
_user_specified_nameinputs
╣
Й
>__inference_bn4_layer_call_and_return_conditional_losses_10274

inputs%
readvariableop_resource:@'
readvariableop_1_resource:@6
(fusedbatchnormv3_readvariableop_resource:@8
*fusedbatchnormv3_readvariableop_1_resource:@
identityИвFusedBatchNormV3/ReadVariableOpв!FusedBatchNormV3/ReadVariableOp_1вReadVariableOpвReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype0Д
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype0И
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype0╚
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+                           @:@:@:@:@:*
epsilon%oГ:*
is_training( }
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+                           @░
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+                           @: : : : 2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2$
ReadVariableOp_1ReadVariableOp_12 
ReadVariableOpReadVariableOp:i e
A
_output_shapes/
-:+                           @
 
_user_specified_nameinputs
л
C
'__inference_flatten_layer_call_fn_11812

inputs
identityо
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         └* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *K
fFRD
B__inference_flatten_layer_call_and_return_conditional_losses_10439a
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:         └"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:         @:W S
/
_output_shapes
:         @
 
_user_specified_nameinputs
е

√
B__inference_conv2D3_layer_call_and_return_conditional_losses_10379

inputs8
conv2d_readvariableop_resource:@@-
biasadd_readvariableop_resource:@
identityИвBiasAdd/ReadVariableOpвConv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype0Щ
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         @*
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
:         @g
IdentityIdentityBiasAdd:output:0^NoOp*
T0*/
_output_shapes
:         @w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:         @: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:         @
 
_user_specified_nameinputs
щ
╛
#__inference_bn2_layer_call_fn_11566

inputs
unknown:@
	unknown_0:@
	unknown_1:@
	unknown_2:@
identityИвStatefulPartitionedCallЕ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+                           @*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *G
fBR@
>__inference_bn2_layer_call_and_return_conditional_losses_10128Й
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+                           @`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+                           @: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+                           @
 
_user_specified_nameinputs
ш
ї
'__inference_encoder_layer_call_fn_11174

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

unknown_22:@

unknown_23:	└


unknown_24:


unknown_25:	└


unknown_26:

identity

identity_1ИвStatefulPartitionedCall╞
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
unknown_26*(
Tin!
2*
Tout
2*
_collective_manager_ids
 *:
_output_shapes(
&:         
:         
*6
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8В *K
fFRD
B__inference_encoder_layer_call_and_return_conditional_losses_10630o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         
q

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*'
_output_shapes
:         
`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "!

identity_1Identity_1:output:0"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*f
_input_shapesU
S:         : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:         
 
_user_specified_nameinputs
ы
╛
#__inference_bn3_layer_call_fn_11670

inputs
unknown:@
	unknown_0:@
	unknown_1:@
	unknown_2:@
identityИвStatefulPartitionedCallЗ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+                           @*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *G
fBR@
>__inference_bn3_layer_call_and_return_conditional_losses_10210Й
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+                           @`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+                           @: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+                           @
 
_user_specified_nameinputs
┘
Ў
#__inference_signature_wrapper_11111
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

unknown_22:@

unknown_23:	└


unknown_24:


unknown_25:	└


unknown_26:

identity

identity_1ИвStatefulPartitionedCall▒
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
unknown_26*(
Tin!
2*
Tout
2*
_collective_manager_ids
 *:
_output_shapes(
&:         
:         
*>
_read_only_resource_inputs 
	
*-
config_proto

CPU

GPU 2J 8В *)
f$R"
 __inference__wrapped_model_10045o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         
q

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*'
_output_shapes
:         
`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "!

identity_1Identity_1:output:0"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*f
_input_shapesU
S:         : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:\ X
/
_output_shapes
:         
%
_user_specified_nameinput_layer
рB
╘	
B__inference_encoder_layer_call_and_return_conditional_losses_10551
input_layer'
conv2d1_10478: 
conv2d1_10480: 
	bn1_10483: 
	bn1_10485: 
	bn1_10487: 
	bn1_10489: '
conv2d2_10493: @
conv2d2_10495:@
	bn2_10498:@
	bn2_10500:@
	bn2_10502:@
	bn2_10504:@'
conv2d3_10508:@@
conv2d3_10510:@
	bn3_10513:@
	bn3_10515:@
	bn3_10517:@
	bn3_10519:@'
conv2d4_10523:@@
conv2d4_10525:@
	bn4_10528:@
	bn4_10530:@
	bn4_10532:@
	bn4_10534:@
logvar_10539:	└

logvar_10541:


mean_10544:	└


mean_10546:

identity

identity_1Ивbn1/StatefulPartitionedCallвbn2/StatefulPartitionedCallвbn3/StatefulPartitionedCallвbn4/StatefulPartitionedCallвconv2D1/StatefulPartitionedCallвconv2D2/StatefulPartitionedCallвconv2D3/StatefulPartitionedCallвconv2D4/StatefulPartitionedCallвlogVar/StatefulPartitionedCallвmean/StatefulPartitionedCallЎ
conv2D1/StatefulPartitionedCallStatefulPartitionedCallinput_layerconv2d1_10478conv2d1_10480*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:          *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *K
fFRD
B__inference_conv2D1_layer_call_and_return_conditional_losses_10315Э
bn1/StatefulPartitionedCallStatefulPartitionedCall(conv2D1/StatefulPartitionedCall:output:0	bn1_10483	bn1_10485	bn1_10487	bn1_10489*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:          *&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *G
fBR@
>__inference_bn1_layer_call_and_return_conditional_losses_10082┘
lReLU1/PartitionedCallPartitionedCall$bn1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:          * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *J
fERC
A__inference_lReLU1_layer_call_and_return_conditional_losses_10335К
conv2D2/StatefulPartitionedCallStatefulPartitionedCalllReLU1/PartitionedCall:output:0conv2d2_10493conv2d2_10495*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         @*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *K
fFRD
B__inference_conv2D2_layer_call_and_return_conditional_losses_10347Э
bn2/StatefulPartitionedCallStatefulPartitionedCall(conv2D2/StatefulPartitionedCall:output:0	bn2_10498	bn2_10500	bn2_10502	bn2_10504*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         @*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *G
fBR@
>__inference_bn2_layer_call_and_return_conditional_losses_10146┘
lReLU2/PartitionedCallPartitionedCall$bn2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         @* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *J
fERC
A__inference_lReLU2_layer_call_and_return_conditional_losses_10367К
conv2D3/StatefulPartitionedCallStatefulPartitionedCalllReLU2/PartitionedCall:output:0conv2d3_10508conv2d3_10510*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         @*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *K
fFRD
B__inference_conv2D3_layer_call_and_return_conditional_losses_10379Э
bn3/StatefulPartitionedCallStatefulPartitionedCall(conv2D3/StatefulPartitionedCall:output:0	bn3_10513	bn3_10515	bn3_10517	bn3_10519*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         @*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *G
fBR@
>__inference_bn3_layer_call_and_return_conditional_losses_10210┘
lReLU3/PartitionedCallPartitionedCall$bn3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         @* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *J
fERC
A__inference_lReLU3_layer_call_and_return_conditional_losses_10399К
conv2D4/StatefulPartitionedCallStatefulPartitionedCalllReLU3/PartitionedCall:output:0conv2d4_10523conv2d4_10525*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         @*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *K
fFRD
B__inference_conv2D4_layer_call_and_return_conditional_losses_10411Э
bn4/StatefulPartitionedCallStatefulPartitionedCall(conv2D4/StatefulPartitionedCall:output:0	bn4_10528	bn4_10530	bn4_10532	bn4_10534*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         @*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *G
fBR@
>__inference_bn4_layer_call_and_return_conditional_losses_10274┘
lReLU4/PartitionedCallPartitionedCall$bn4/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         @* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *J
fERC
A__inference_lReLU4_layer_call_and_return_conditional_losses_10431╧
flatten/PartitionedCallPartitionedCalllReLU4/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         └* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *K
fFRD
B__inference_flatten_layer_call_and_return_conditional_losses_10439 
logVar/StatefulPartitionedCallStatefulPartitionedCall flatten/PartitionedCall:output:0logvar_10539logvar_10541*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         
*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *J
fERC
A__inference_logVar_layer_call_and_return_conditional_losses_10451ў
mean/StatefulPartitionedCallStatefulPartitionedCall flatten/PartitionedCall:output:0
mean_10544
mean_10546*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         
*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *H
fCRA
?__inference_mean_layer_call_and_return_conditional_losses_10467t
IdentityIdentity%mean/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         
x

Identity_1Identity'logVar/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         
Ж
NoOpNoOp^bn1/StatefulPartitionedCall^bn2/StatefulPartitionedCall^bn3/StatefulPartitionedCall^bn4/StatefulPartitionedCall ^conv2D1/StatefulPartitionedCall ^conv2D2/StatefulPartitionedCall ^conv2D3/StatefulPartitionedCall ^conv2D4/StatefulPartitionedCall^logVar/StatefulPartitionedCall^mean/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "!

identity_1Identity_1:output:0"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*f
_input_shapesU
S:         : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2:
bn1/StatefulPartitionedCallbn1/StatefulPartitionedCall2:
bn2/StatefulPartitionedCallbn2/StatefulPartitionedCall2:
bn3/StatefulPartitionedCallbn3/StatefulPartitionedCall2:
bn4/StatefulPartitionedCallbn4/StatefulPartitionedCall2B
conv2D1/StatefulPartitionedCallconv2D1/StatefulPartitionedCall2B
conv2D2/StatefulPartitionedCallconv2D2/StatefulPartitionedCall2B
conv2D3/StatefulPartitionedCallconv2D3/StatefulPartitionedCall2B
conv2D4/StatefulPartitionedCallconv2D4/StatefulPartitionedCall2@
logVar/StatefulPartitionedCalllogVar/StatefulPartitionedCall2<
mean/StatefulPartitionedCallmean/StatefulPartitionedCall:\ X
/
_output_shapes
:         
%
_user_specified_nameinput_layer
ц
Ь
'__inference_conv2D4_layer_call_fn_11725

inputs!
unknown:@@
	unknown_0:@
identityИвStatefulPartitionedCall▀
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         @*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *K
fFRD
B__inference_conv2D4_layer_call_and_return_conditional_losses_10411w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:         @`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:         @: : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:         @
 
_user_specified_nameinputs
╤B
╧	
B__inference_encoder_layer_call_and_return_conditional_losses_10769

inputs'
conv2d1_10696: 
conv2d1_10698: 
	bn1_10701: 
	bn1_10703: 
	bn1_10705: 
	bn1_10707: '
conv2d2_10711: @
conv2d2_10713:@
	bn2_10716:@
	bn2_10718:@
	bn2_10720:@
	bn2_10722:@'
conv2d3_10726:@@
conv2d3_10728:@
	bn3_10731:@
	bn3_10733:@
	bn3_10735:@
	bn3_10737:@'
conv2d4_10741:@@
conv2d4_10743:@
	bn4_10746:@
	bn4_10748:@
	bn4_10750:@
	bn4_10752:@
logvar_10757:	└

logvar_10759:


mean_10762:	└


mean_10764:

identity

identity_1Ивbn1/StatefulPartitionedCallвbn2/StatefulPartitionedCallвbn3/StatefulPartitionedCallвbn4/StatefulPartitionedCallвconv2D1/StatefulPartitionedCallвconv2D2/StatefulPartitionedCallвconv2D3/StatefulPartitionedCallвconv2D4/StatefulPartitionedCallвlogVar/StatefulPartitionedCallвmean/StatefulPartitionedCallё
conv2D1/StatefulPartitionedCallStatefulPartitionedCallinputsconv2d1_10696conv2d1_10698*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:          *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *K
fFRD
B__inference_conv2D1_layer_call_and_return_conditional_losses_10315Э
bn1/StatefulPartitionedCallStatefulPartitionedCall(conv2D1/StatefulPartitionedCall:output:0	bn1_10701	bn1_10703	bn1_10705	bn1_10707*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:          *&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *G
fBR@
>__inference_bn1_layer_call_and_return_conditional_losses_10082┘
lReLU1/PartitionedCallPartitionedCall$bn1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:          * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *J
fERC
A__inference_lReLU1_layer_call_and_return_conditional_losses_10335К
conv2D2/StatefulPartitionedCallStatefulPartitionedCalllReLU1/PartitionedCall:output:0conv2d2_10711conv2d2_10713*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         @*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *K
fFRD
B__inference_conv2D2_layer_call_and_return_conditional_losses_10347Э
bn2/StatefulPartitionedCallStatefulPartitionedCall(conv2D2/StatefulPartitionedCall:output:0	bn2_10716	bn2_10718	bn2_10720	bn2_10722*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         @*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *G
fBR@
>__inference_bn2_layer_call_and_return_conditional_losses_10146┘
lReLU2/PartitionedCallPartitionedCall$bn2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         @* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *J
fERC
A__inference_lReLU2_layer_call_and_return_conditional_losses_10367К
conv2D3/StatefulPartitionedCallStatefulPartitionedCalllReLU2/PartitionedCall:output:0conv2d3_10726conv2d3_10728*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         @*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *K
fFRD
B__inference_conv2D3_layer_call_and_return_conditional_losses_10379Э
bn3/StatefulPartitionedCallStatefulPartitionedCall(conv2D3/StatefulPartitionedCall:output:0	bn3_10731	bn3_10733	bn3_10735	bn3_10737*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         @*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *G
fBR@
>__inference_bn3_layer_call_and_return_conditional_losses_10210┘
lReLU3/PartitionedCallPartitionedCall$bn3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         @* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *J
fERC
A__inference_lReLU3_layer_call_and_return_conditional_losses_10399К
conv2D4/StatefulPartitionedCallStatefulPartitionedCalllReLU3/PartitionedCall:output:0conv2d4_10741conv2d4_10743*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         @*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *K
fFRD
B__inference_conv2D4_layer_call_and_return_conditional_losses_10411Э
bn4/StatefulPartitionedCallStatefulPartitionedCall(conv2D4/StatefulPartitionedCall:output:0	bn4_10746	bn4_10748	bn4_10750	bn4_10752*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         @*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *G
fBR@
>__inference_bn4_layer_call_and_return_conditional_losses_10274┘
lReLU4/PartitionedCallPartitionedCall$bn4/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         @* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *J
fERC
A__inference_lReLU4_layer_call_and_return_conditional_losses_10431╧
flatten/PartitionedCallPartitionedCalllReLU4/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         └* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *K
fFRD
B__inference_flatten_layer_call_and_return_conditional_losses_10439 
logVar/StatefulPartitionedCallStatefulPartitionedCall flatten/PartitionedCall:output:0logvar_10757logvar_10759*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         
*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *J
fERC
A__inference_logVar_layer_call_and_return_conditional_losses_10451ў
mean/StatefulPartitionedCallStatefulPartitionedCall flatten/PartitionedCall:output:0
mean_10762
mean_10764*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         
*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *H
fCRA
?__inference_mean_layer_call_and_return_conditional_losses_10467t
IdentityIdentity%mean/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         
x

Identity_1Identity'logVar/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         
Ж
NoOpNoOp^bn1/StatefulPartitionedCall^bn2/StatefulPartitionedCall^bn3/StatefulPartitionedCall^bn4/StatefulPartitionedCall ^conv2D1/StatefulPartitionedCall ^conv2D2/StatefulPartitionedCall ^conv2D3/StatefulPartitionedCall ^conv2D4/StatefulPartitionedCall^logVar/StatefulPartitionedCall^mean/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "!

identity_1Identity_1:output:0"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*f
_input_shapesU
S:         : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2:
bn1/StatefulPartitionedCallbn1/StatefulPartitionedCall2:
bn2/StatefulPartitionedCallbn2/StatefulPartitionedCall2:
bn3/StatefulPartitionedCallbn3/StatefulPartitionedCall2:
bn4/StatefulPartitionedCallbn4/StatefulPartitionedCall2B
conv2D1/StatefulPartitionedCallconv2D1/StatefulPartitionedCall2B
conv2D2/StatefulPartitionedCallconv2D2/StatefulPartitionedCall2B
conv2D3/StatefulPartitionedCallconv2D3/StatefulPartitionedCall2B
conv2D4/StatefulPartitionedCallconv2D4/StatefulPartitionedCall2@
logVar/StatefulPartitionedCalllogVar/StatefulPartitionedCall2<
mean/StatefulPartitionedCallmean/StatefulPartitionedCall:W S
/
_output_shapes
:         
 
_user_specified_nameinputs
╞	
ё
?__inference_mean_layer_call_and_return_conditional_losses_11837

inputs1
matmul_readvariableop_resource:	└
-
biasadd_readvariableop_resource:

identityИвBiasAdd/ReadVariableOpвMatMul/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	└
*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:
*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         
_
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:         
w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:         └: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:         └
 
_user_specified_nameinputs
е

√
B__inference_conv2D4_layer_call_and_return_conditional_losses_10411

inputs8
conv2d_readvariableop_resource:@@-
biasadd_readvariableop_resource:@
identityИвBiasAdd/ReadVariableOpвConv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype0Щ
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         @*
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
:         @g
IdentityIdentityBiasAdd:output:0^NoOp*
T0*/
_output_shapes
:         @w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:         @: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:         @
 
_user_specified_nameinputs
ц
Ь
'__inference_conv2D3_layer_call_fn_11634

inputs!
unknown:@@
	unknown_0:@
identityИвStatefulPartitionedCall▀
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         @*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *K
fFRD
B__inference_conv2D3_layer_call_and_return_conditional_losses_10379w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:         @`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:         @: : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:         @
 
_user_specified_nameinputs
√
]
A__inference_lReLU3_layer_call_and_return_conditional_losses_11716

inputs
identity_
	LeakyRelu	LeakyReluinputs*/
_output_shapes
:         @*
alpha%ЪЩЩ>g
IdentityIdentityLeakyRelu:activations:0*
T0*/
_output_shapes
:         @"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:         @:W S
/
_output_shapes
:         @
 
_user_specified_nameinputs
щ
╛
#__inference_bn3_layer_call_fn_11657

inputs
unknown:@
	unknown_0:@
	unknown_1:@
	unknown_2:@
identityИвStatefulPartitionedCallЕ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+                           @*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *G
fBR@
>__inference_bn3_layer_call_and_return_conditional_losses_10192Й
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+                           @`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+                           @: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+                           @
 
_user_specified_nameinputs
─
^
B__inference_flatten_layer_call_and_return_conditional_losses_11818

inputs
identityV
ConstConst*
_output_shapes
:*
dtype0*
valueB"    @  ]
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:         └Y
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:         └"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:         @:W S
/
_output_shapes
:         @
 
_user_specified_nameinputs
╔B
╧	
B__inference_encoder_layer_call_and_return_conditional_losses_10630

inputs'
conv2d1_10557: 
conv2d1_10559: 
	bn1_10562: 
	bn1_10564: 
	bn1_10566: 
	bn1_10568: '
conv2d2_10572: @
conv2d2_10574:@
	bn2_10577:@
	bn2_10579:@
	bn2_10581:@
	bn2_10583:@'
conv2d3_10587:@@
conv2d3_10589:@
	bn3_10592:@
	bn3_10594:@
	bn3_10596:@
	bn3_10598:@'
conv2d4_10602:@@
conv2d4_10604:@
	bn4_10607:@
	bn4_10609:@
	bn4_10611:@
	bn4_10613:@
logvar_10618:	└

logvar_10620:


mean_10623:	└


mean_10625:

identity

identity_1Ивbn1/StatefulPartitionedCallвbn2/StatefulPartitionedCallвbn3/StatefulPartitionedCallвbn4/StatefulPartitionedCallвconv2D1/StatefulPartitionedCallвconv2D2/StatefulPartitionedCallвconv2D3/StatefulPartitionedCallвconv2D4/StatefulPartitionedCallвlogVar/StatefulPartitionedCallвmean/StatefulPartitionedCallё
conv2D1/StatefulPartitionedCallStatefulPartitionedCallinputsconv2d1_10557conv2d1_10559*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:          *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *K
fFRD
B__inference_conv2D1_layer_call_and_return_conditional_losses_10315Ы
bn1/StatefulPartitionedCallStatefulPartitionedCall(conv2D1/StatefulPartitionedCall:output:0	bn1_10562	bn1_10564	bn1_10566	bn1_10568*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:          *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *G
fBR@
>__inference_bn1_layer_call_and_return_conditional_losses_10064┘
lReLU1/PartitionedCallPartitionedCall$bn1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:          * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *J
fERC
A__inference_lReLU1_layer_call_and_return_conditional_losses_10335К
conv2D2/StatefulPartitionedCallStatefulPartitionedCalllReLU1/PartitionedCall:output:0conv2d2_10572conv2d2_10574*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         @*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *K
fFRD
B__inference_conv2D2_layer_call_and_return_conditional_losses_10347Ы
bn2/StatefulPartitionedCallStatefulPartitionedCall(conv2D2/StatefulPartitionedCall:output:0	bn2_10577	bn2_10579	bn2_10581	bn2_10583*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         @*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *G
fBR@
>__inference_bn2_layer_call_and_return_conditional_losses_10128┘
lReLU2/PartitionedCallPartitionedCall$bn2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         @* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *J
fERC
A__inference_lReLU2_layer_call_and_return_conditional_losses_10367К
conv2D3/StatefulPartitionedCallStatefulPartitionedCalllReLU2/PartitionedCall:output:0conv2d3_10587conv2d3_10589*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         @*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *K
fFRD
B__inference_conv2D3_layer_call_and_return_conditional_losses_10379Ы
bn3/StatefulPartitionedCallStatefulPartitionedCall(conv2D3/StatefulPartitionedCall:output:0	bn3_10592	bn3_10594	bn3_10596	bn3_10598*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         @*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *G
fBR@
>__inference_bn3_layer_call_and_return_conditional_losses_10192┘
lReLU3/PartitionedCallPartitionedCall$bn3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         @* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *J
fERC
A__inference_lReLU3_layer_call_and_return_conditional_losses_10399К
conv2D4/StatefulPartitionedCallStatefulPartitionedCalllReLU3/PartitionedCall:output:0conv2d4_10602conv2d4_10604*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         @*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *K
fFRD
B__inference_conv2D4_layer_call_and_return_conditional_losses_10411Ы
bn4/StatefulPartitionedCallStatefulPartitionedCall(conv2D4/StatefulPartitionedCall:output:0	bn4_10607	bn4_10609	bn4_10611	bn4_10613*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         @*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *G
fBR@
>__inference_bn4_layer_call_and_return_conditional_losses_10256┘
lReLU4/PartitionedCallPartitionedCall$bn4/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         @* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *J
fERC
A__inference_lReLU4_layer_call_and_return_conditional_losses_10431╧
flatten/PartitionedCallPartitionedCalllReLU4/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         └* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *K
fFRD
B__inference_flatten_layer_call_and_return_conditional_losses_10439 
logVar/StatefulPartitionedCallStatefulPartitionedCall flatten/PartitionedCall:output:0logvar_10618logvar_10620*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         
*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *J
fERC
A__inference_logVar_layer_call_and_return_conditional_losses_10451ў
mean/StatefulPartitionedCallStatefulPartitionedCall flatten/PartitionedCall:output:0
mean_10623
mean_10625*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         
*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *H
fCRA
?__inference_mean_layer_call_and_return_conditional_losses_10467t
IdentityIdentity%mean/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         
x

Identity_1Identity'logVar/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         
Ж
NoOpNoOp^bn1/StatefulPartitionedCall^bn2/StatefulPartitionedCall^bn3/StatefulPartitionedCall^bn4/StatefulPartitionedCall ^conv2D1/StatefulPartitionedCall ^conv2D2/StatefulPartitionedCall ^conv2D3/StatefulPartitionedCall ^conv2D4/StatefulPartitionedCall^logVar/StatefulPartitionedCall^mean/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "!

identity_1Identity_1:output:0"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*f
_input_shapesU
S:         : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2:
bn1/StatefulPartitionedCallbn1/StatefulPartitionedCall2:
bn2/StatefulPartitionedCallbn2/StatefulPartitionedCall2:
bn3/StatefulPartitionedCallbn3/StatefulPartitionedCall2:
bn4/StatefulPartitionedCallbn4/StatefulPartitionedCall2B
conv2D1/StatefulPartitionedCallconv2D1/StatefulPartitionedCall2B
conv2D2/StatefulPartitionedCallconv2D2/StatefulPartitionedCall2B
conv2D3/StatefulPartitionedCallconv2D3/StatefulPartitionedCall2B
conv2D4/StatefulPartitionedCallconv2D4/StatefulPartitionedCall2@
logVar/StatefulPartitionedCalllogVar/StatefulPartitionedCall2<
mean/StatefulPartitionedCallmean/StatefulPartitionedCall:W S
/
_output_shapes
:         
 
_user_specified_nameinputs
╚	
є
A__inference_logVar_layer_call_and_return_conditional_losses_11856

inputs1
matmul_readvariableop_resource:	└
-
biasadd_readvariableop_resource:

identityИвBiasAdd/ReadVariableOpвMatMul/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	└
*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:
*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         
_
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:         
w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:         └: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:         └
 
_user_specified_nameinputs
√
]
A__inference_lReLU2_layer_call_and_return_conditional_losses_10367

inputs
identity_
	LeakyRelu	LeakyReluinputs*/
_output_shapes
:         @*
alpha%ЪЩЩ>g
IdentityIdentityLeakyRelu:activations:0*
T0*/
_output_shapes
:         @"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:         @:W S
/
_output_shapes
:         @
 
_user_specified_nameinputs
╗
Т
$__inference_mean_layer_call_fn_11827

inputs
unknown:	└

	unknown_0:

identityИвStatefulPartitionedCall╘
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         
*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *H
fCRA
?__inference_mean_layer_call_and_return_conditional_losses_10467o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         
`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:         └: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:         └
 
_user_specified_nameinputs
╣
Й
>__inference_bn1_layer_call_and_return_conditional_losses_11524

inputs%
readvariableop_resource: '
readvariableop_1_resource: 6
(fusedbatchnormv3_readvariableop_resource: 8
*fusedbatchnormv3_readvariableop_1_resource: 
identityИвFusedBatchNormV3/ReadVariableOpв!FusedBatchNormV3/ReadVariableOp_1вReadVariableOpвReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype0Д
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype0И
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype0╚
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+                            : : : : :*
epsilon%oГ:*
is_training( }
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+                            ░
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+                            : : : : 2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2$
ReadVariableOp_1ReadVariableOp_12 
ReadVariableOpReadVariableOp:i e
A
_output_shapes/
-:+                            
 
_user_specified_nameinputs
╪B
╘	
B__inference_encoder_layer_call_and_return_conditional_losses_10475
input_layer'
conv2d1_10316: 
conv2d1_10318: 
	bn1_10321: 
	bn1_10323: 
	bn1_10325: 
	bn1_10327: '
conv2d2_10348: @
conv2d2_10350:@
	bn2_10353:@
	bn2_10355:@
	bn2_10357:@
	bn2_10359:@'
conv2d3_10380:@@
conv2d3_10382:@
	bn3_10385:@
	bn3_10387:@
	bn3_10389:@
	bn3_10391:@'
conv2d4_10412:@@
conv2d4_10414:@
	bn4_10417:@
	bn4_10419:@
	bn4_10421:@
	bn4_10423:@
logvar_10452:	└

logvar_10454:


mean_10468:	└


mean_10470:

identity

identity_1Ивbn1/StatefulPartitionedCallвbn2/StatefulPartitionedCallвbn3/StatefulPartitionedCallвbn4/StatefulPartitionedCallвconv2D1/StatefulPartitionedCallвconv2D2/StatefulPartitionedCallвconv2D3/StatefulPartitionedCallвconv2D4/StatefulPartitionedCallвlogVar/StatefulPartitionedCallвmean/StatefulPartitionedCallЎ
conv2D1/StatefulPartitionedCallStatefulPartitionedCallinput_layerconv2d1_10316conv2d1_10318*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:          *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *K
fFRD
B__inference_conv2D1_layer_call_and_return_conditional_losses_10315Ы
bn1/StatefulPartitionedCallStatefulPartitionedCall(conv2D1/StatefulPartitionedCall:output:0	bn1_10321	bn1_10323	bn1_10325	bn1_10327*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:          *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *G
fBR@
>__inference_bn1_layer_call_and_return_conditional_losses_10064┘
lReLU1/PartitionedCallPartitionedCall$bn1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:          * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *J
fERC
A__inference_lReLU1_layer_call_and_return_conditional_losses_10335К
conv2D2/StatefulPartitionedCallStatefulPartitionedCalllReLU1/PartitionedCall:output:0conv2d2_10348conv2d2_10350*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         @*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *K
fFRD
B__inference_conv2D2_layer_call_and_return_conditional_losses_10347Ы
bn2/StatefulPartitionedCallStatefulPartitionedCall(conv2D2/StatefulPartitionedCall:output:0	bn2_10353	bn2_10355	bn2_10357	bn2_10359*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         @*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *G
fBR@
>__inference_bn2_layer_call_and_return_conditional_losses_10128┘
lReLU2/PartitionedCallPartitionedCall$bn2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         @* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *J
fERC
A__inference_lReLU2_layer_call_and_return_conditional_losses_10367К
conv2D3/StatefulPartitionedCallStatefulPartitionedCalllReLU2/PartitionedCall:output:0conv2d3_10380conv2d3_10382*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         @*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *K
fFRD
B__inference_conv2D3_layer_call_and_return_conditional_losses_10379Ы
bn3/StatefulPartitionedCallStatefulPartitionedCall(conv2D3/StatefulPartitionedCall:output:0	bn3_10385	bn3_10387	bn3_10389	bn3_10391*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         @*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *G
fBR@
>__inference_bn3_layer_call_and_return_conditional_losses_10192┘
lReLU3/PartitionedCallPartitionedCall$bn3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         @* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *J
fERC
A__inference_lReLU3_layer_call_and_return_conditional_losses_10399К
conv2D4/StatefulPartitionedCallStatefulPartitionedCalllReLU3/PartitionedCall:output:0conv2d4_10412conv2d4_10414*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         @*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *K
fFRD
B__inference_conv2D4_layer_call_and_return_conditional_losses_10411Ы
bn4/StatefulPartitionedCallStatefulPartitionedCall(conv2D4/StatefulPartitionedCall:output:0	bn4_10417	bn4_10419	bn4_10421	bn4_10423*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         @*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *G
fBR@
>__inference_bn4_layer_call_and_return_conditional_losses_10256┘
lReLU4/PartitionedCallPartitionedCall$bn4/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         @* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *J
fERC
A__inference_lReLU4_layer_call_and_return_conditional_losses_10431╧
flatten/PartitionedCallPartitionedCalllReLU4/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         └* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *K
fFRD
B__inference_flatten_layer_call_and_return_conditional_losses_10439 
logVar/StatefulPartitionedCallStatefulPartitionedCall flatten/PartitionedCall:output:0logvar_10452logvar_10454*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         
*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *J
fERC
A__inference_logVar_layer_call_and_return_conditional_losses_10451ў
mean/StatefulPartitionedCallStatefulPartitionedCall flatten/PartitionedCall:output:0
mean_10468
mean_10470*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         
*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *H
fCRA
?__inference_mean_layer_call_and_return_conditional_losses_10467t
IdentityIdentity%mean/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         
x

Identity_1Identity'logVar/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         
Ж
NoOpNoOp^bn1/StatefulPartitionedCall^bn2/StatefulPartitionedCall^bn3/StatefulPartitionedCall^bn4/StatefulPartitionedCall ^conv2D1/StatefulPartitionedCall ^conv2D2/StatefulPartitionedCall ^conv2D3/StatefulPartitionedCall ^conv2D4/StatefulPartitionedCall^logVar/StatefulPartitionedCall^mean/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "!

identity_1Identity_1:output:0"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*f
_input_shapesU
S:         : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2:
bn1/StatefulPartitionedCallbn1/StatefulPartitionedCall2:
bn2/StatefulPartitionedCallbn2/StatefulPartitionedCall2:
bn3/StatefulPartitionedCallbn3/StatefulPartitionedCall2:
bn4/StatefulPartitionedCallbn4/StatefulPartitionedCall2B
conv2D1/StatefulPartitionedCallconv2D1/StatefulPartitionedCall2B
conv2D2/StatefulPartitionedCallconv2D2/StatefulPartitionedCall2B
conv2D3/StatefulPartitionedCallconv2D3/StatefulPartitionedCall2B
conv2D4/StatefulPartitionedCallconv2D4/StatefulPartitionedCall2@
logVar/StatefulPartitionedCalllogVar/StatefulPartitionedCall2<
mean/StatefulPartitionedCallmean/StatefulPartitionedCall:\ X
/
_output_shapes
:         
%
_user_specified_nameinput_layer
ц
Ь
'__inference_conv2D2_layer_call_fn_11543

inputs!
unknown: @
	unknown_0:@
identityИвStatefulPartitionedCall▀
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         @*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *K
fFRD
B__inference_conv2D2_layer_call_and_return_conditional_losses_10347w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:         @`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:          : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:          
 
_user_specified_nameinputs
 
·
'__inference_encoder_layer_call_fn_10830
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

unknown_22:@

unknown_23:	└


unknown_24:


unknown_25:	└


unknown_26:

identity

identity_1ИвStatefulPartitionedCall╙
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
unknown_26*(
Tin!
2*
Tout
2*
_collective_manager_ids
 *:
_output_shapes(
&:         
:         
*>
_read_only_resource_inputs 
	
*-
config_proto

CPU

GPU 2J 8В *K
fFRD
B__inference_encoder_layer_call_and_return_conditional_losses_10769o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         
q

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*'
_output_shapes
:         
`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "!

identity_1Identity_1:output:0"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*f
_input_shapesU
S:         : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:\ X
/
_output_shapes
:         
%
_user_specified_nameinput_layer
╞	
ё
?__inference_mean_layer_call_and_return_conditional_losses_10467

inputs1
matmul_readvariableop_resource:	└
-
biasadd_readvariableop_resource:

identityИвBiasAdd/ReadVariableOpвMatMul/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	└
*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:
*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         
_
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:         
w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:         └: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:         └
 
_user_specified_nameinputs
е

√
B__inference_conv2D2_layer_call_and_return_conditional_losses_11553

inputs8
conv2d_readvariableop_resource: @-
biasadd_readvariableop_resource:@
identityИвBiasAdd/ReadVariableOpвConv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: @*
dtype0Щ
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         @*
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
:         @g
IdentityIdentityBiasAdd:output:0^NoOp*
T0*/
_output_shapes
:         @w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:          : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:          
 
_user_specified_nameinputs
╖
B
&__inference_lReLU4_layer_call_fn_11802

inputs
identity┤
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         @* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *J
fERC
A__inference_lReLU4_layer_call_and_return_conditional_losses_10431h
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:         @"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:         @:W S
/
_output_shapes
:         @
 
_user_specified_nameinputs
╚	
є
A__inference_logVar_layer_call_and_return_conditional_losses_10451

inputs1
matmul_readvariableop_resource:	└
-
biasadd_readvariableop_resource:

identityИвBiasAdd/ReadVariableOpвMatMul/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	└
*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:
*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         
_
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:         
w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:         └: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:         └
 
_user_specified_nameinputs
╣
Й
>__inference_bn1_layer_call_and_return_conditional_losses_10082

inputs%
readvariableop_resource: '
readvariableop_1_resource: 6
(fusedbatchnormv3_readvariableop_resource: 8
*fusedbatchnormv3_readvariableop_1_resource: 
identityИвFusedBatchNormV3/ReadVariableOpв!FusedBatchNormV3/ReadVariableOp_1вReadVariableOpвReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype0Д
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype0И
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype0╚
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+                            : : : : :*
epsilon%oГ:*
is_training( }
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+                            ░
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+                            : : : : 2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2$
ReadVariableOp_1ReadVariableOp_12 
ReadVariableOpReadVariableOp:i e
A
_output_shapes/
-:+                            
 
_user_specified_nameinputs
е

√
B__inference_conv2D3_layer_call_and_return_conditional_losses_11644

inputs8
conv2d_readvariableop_resource:@@-
biasadd_readvariableop_resource:@
identityИвBiasAdd/ReadVariableOpвConv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype0Щ
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         @*
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
:         @g
IdentityIdentityBiasAdd:output:0^NoOp*
T0*/
_output_shapes
:         @w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:         @: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:         @
 
_user_specified_nameinputs
ц
Ь
'__inference_conv2D1_layer_call_fn_11452

inputs!
unknown: 
	unknown_0: 
identityИвStatefulPartitionedCall▀
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:          *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *K
fFRD
B__inference_conv2D1_layer_call_and_return_conditional_losses_10315w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:          `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:         : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:         
 
_user_specified_nameinputs
√
]
A__inference_lReLU2_layer_call_and_return_conditional_losses_11625

inputs
identity_
	LeakyRelu	LeakyReluinputs*/
_output_shapes
:         @*
alpha%ЪЩЩ>g
IdentityIdentityLeakyRelu:activations:0*
T0*/
_output_shapes
:         @"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:         @:W S
/
_output_shapes
:         @
 
_user_specified_nameinputs
є
н
>__inference_bn2_layer_call_and_return_conditional_losses_11597

inputs%
readvariableop_resource:@'
readvariableop_1_resource:@6
(fusedbatchnormv3_readvariableop_resource:@8
*fusedbatchnormv3_readvariableop_1_resource:@
identityИвAssignNewValueвAssignNewValue_1вFusedBatchNormV3/ReadVariableOpв!FusedBatchNormV3/ReadVariableOp_1вReadVariableOpвReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype0Д
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype0И
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype0╓
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+                           @:@:@:@:@:*
epsilon%oГ:*
exponential_avg_factor%
╫#<╞
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(╨
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(}
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+                           @╘
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+                           @: : : : 2$
AssignNewValue_1AssignNewValue_12 
AssignNewValueAssignNewValue2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2$
ReadVariableOp_1ReadVariableOp_12 
ReadVariableOpReadVariableOp:i e
A
_output_shapes/
-:+                           @
 
_user_specified_nameinputs"є
L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*є
serving_default▀
K
input_layer<
serving_default_input_layer:0         :
logVar0
StatefulPartitionedCall:0         
8
mean0
StatefulPartitionedCall:1         
tensorflow/serving/predict:ец
о
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
layer-13
layer_with_weights-8
layer-14
layer_with_weights-9
layer-15
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
_default_save_signature

signatures"
_tf_keras_network
"
_tf_keras_input_layer
▌
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses

kernel
 bias
 !_jit_compiled_convolution_op"
_tf_keras_layer
ъ
"	variables
#trainable_variables
$regularization_losses
%	keras_api
&__call__
*'&call_and_return_all_conditional_losses
(axis
	)gamma
*beta
+moving_mean
,moving_variance"
_tf_keras_layer
е
-	variables
.trainable_variables
/regularization_losses
0	keras_api
1__call__
*2&call_and_return_all_conditional_losses"
_tf_keras_layer
▌
3	variables
4trainable_variables
5regularization_losses
6	keras_api
7__call__
*8&call_and_return_all_conditional_losses

9kernel
:bias
 ;_jit_compiled_convolution_op"
_tf_keras_layer
ъ
<	variables
=trainable_variables
>regularization_losses
?	keras_api
@__call__
*A&call_and_return_all_conditional_losses
Baxis
	Cgamma
Dbeta
Emoving_mean
Fmoving_variance"
_tf_keras_layer
е
G	variables
Htrainable_variables
Iregularization_losses
J	keras_api
K__call__
*L&call_and_return_all_conditional_losses"
_tf_keras_layer
▌
M	variables
Ntrainable_variables
Oregularization_losses
P	keras_api
Q__call__
*R&call_and_return_all_conditional_losses

Skernel
Tbias
 U_jit_compiled_convolution_op"
_tf_keras_layer
ъ
V	variables
Wtrainable_variables
Xregularization_losses
Y	keras_api
Z__call__
*[&call_and_return_all_conditional_losses
\axis
	]gamma
^beta
_moving_mean
`moving_variance"
_tf_keras_layer
е
a	variables
btrainable_variables
cregularization_losses
d	keras_api
e__call__
*f&call_and_return_all_conditional_losses"
_tf_keras_layer
▌
g	variables
htrainable_variables
iregularization_losses
j	keras_api
k__call__
*l&call_and_return_all_conditional_losses

mkernel
nbias
 o_jit_compiled_convolution_op"
_tf_keras_layer
ъ
p	variables
qtrainable_variables
rregularization_losses
s	keras_api
t__call__
*u&call_and_return_all_conditional_losses
vaxis
	wgamma
xbeta
ymoving_mean
zmoving_variance"
_tf_keras_layer
ж
{	variables
|trainable_variables
}regularization_losses
~	keras_api
__call__
+А&call_and_return_all_conditional_losses"
_tf_keras_layer
л
Б	variables
Вtrainable_variables
Гregularization_losses
Д	keras_api
Е__call__
+Ж&call_and_return_all_conditional_losses"
_tf_keras_layer
├
З	variables
Иtrainable_variables
Йregularization_losses
К	keras_api
Л__call__
+М&call_and_return_all_conditional_losses
Нkernel
	Оbias"
_tf_keras_layer
├
П	variables
Рtrainable_variables
Сregularization_losses
Т	keras_api
У__call__
+Ф&call_and_return_all_conditional_losses
Хkernel
	Цbias"
_tf_keras_layer
·
0
 1
)2
*3
+4
,5
96
:7
C8
D9
E10
F11
S12
T13
]14
^15
_16
`17
m18
n19
w20
x21
y22
z23
Н24
О25
Х26
Ц27"
trackable_list_wrapper
║
0
 1
)2
*3
94
:5
C6
D7
S8
T9
]10
^11
m12
n13
w14
x15
Н16
О17
Х18
Ц19"
trackable_list_wrapper
 "
trackable_list_wrapper
╧
Чnon_trainable_variables
Шlayers
Щmetrics
 Ъlayer_regularization_losses
Ыlayer_metrics
	variables
trainable_variables
regularization_losses
__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
╧
Ьtrace_0
Эtrace_1
Юtrace_2
Яtrace_32▄
'__inference_encoder_layer_call_fn_10691
'__inference_encoder_layer_call_fn_10830
'__inference_encoder_layer_call_fn_11174
'__inference_encoder_layer_call_fn_11237╡
о▓к
FullArgSpec)
args!Ъ
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsв
p 

 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 zЬtrace_0zЭtrace_1zЮtrace_2zЯtrace_3
╗
аtrace_0
бtrace_1
вtrace_2
гtrace_32╚
B__inference_encoder_layer_call_and_return_conditional_losses_10475
B__inference_encoder_layer_call_and_return_conditional_losses_10551
B__inference_encoder_layer_call_and_return_conditional_losses_11340
B__inference_encoder_layer_call_and_return_conditional_losses_11443╡
о▓к
FullArgSpec)
args!Ъ
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsв
p 

 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 zаtrace_0zбtrace_1zвtrace_2zгtrace_3
╧B╠
 __inference__wrapped_model_10045input_layer"Ш
С▓Н
FullArgSpec
argsЪ 
varargsjargs
varkwjkwargs
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
-
дserving_default"
signature_map
.
0
 1"
trackable_list_wrapper
.
0
 1"
trackable_list_wrapper
 "
trackable_list_wrapper
▓
еnon_trainable_variables
жlayers
зmetrics
 иlayer_regularization_losses
йlayer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
у
кtrace_02─
'__inference_conv2D1_layer_call_fn_11452Ш
С▓Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 zкtrace_0
■
лtrace_02▀
B__inference_conv2D1_layer_call_and_return_conditional_losses_11462Ш
С▓Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 zлtrace_0
(:& 2conv2D1/kernel
: 2conv2D1/bias
к2зд
Ы▓Ч
FullArgSpec
argsЪ
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 0
<
)0
*1
+2
,3"
trackable_list_wrapper
.
)0
*1"
trackable_list_wrapper
 "
trackable_list_wrapper
▓
мnon_trainable_variables
нlayers
оmetrics
 пlayer_regularization_losses
░layer_metrics
"	variables
#trainable_variables
$regularization_losses
&__call__
*'&call_and_return_all_conditional_losses
&'"call_and_return_conditional_losses"
_generic_user_object
╜
▒trace_0
▓trace_12В
#__inference_bn1_layer_call_fn_11475
#__inference_bn1_layer_call_fn_11488╡
о▓к
FullArgSpec)
args!Ъ
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsв
p 

 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 z▒trace_0z▓trace_1
є
│trace_0
┤trace_12╕
>__inference_bn1_layer_call_and_return_conditional_losses_11506
>__inference_bn1_layer_call_and_return_conditional_losses_11524╡
о▓к
FullArgSpec)
args!Ъ
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsв
p 

 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 z│trace_0z┤trace_1
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
▓
╡non_trainable_variables
╢layers
╖metrics
 ╕layer_regularization_losses
╣layer_metrics
-	variables
.trainable_variables
/regularization_losses
1__call__
*2&call_and_return_all_conditional_losses
&2"call_and_return_conditional_losses"
_generic_user_object
т
║trace_02├
&__inference_lReLU1_layer_call_fn_11529Ш
С▓Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 z║trace_0
¤
╗trace_02▐
A__inference_lReLU1_layer_call_and_return_conditional_losses_11534Ш
С▓Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 z╗trace_0
.
90
:1"
trackable_list_wrapper
.
90
:1"
trackable_list_wrapper
 "
trackable_list_wrapper
▓
╝non_trainable_variables
╜layers
╛metrics
 ┐layer_regularization_losses
└layer_metrics
3	variables
4trainable_variables
5regularization_losses
7__call__
*8&call_and_return_all_conditional_losses
&8"call_and_return_conditional_losses"
_generic_user_object
у
┴trace_02─
'__inference_conv2D2_layer_call_fn_11543Ш
С▓Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 z┴trace_0
■
┬trace_02▀
B__inference_conv2D2_layer_call_and_return_conditional_losses_11553Ш
С▓Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 z┬trace_0
(:& @2conv2D2/kernel
:@2conv2D2/bias
к2зд
Ы▓Ч
FullArgSpec
argsЪ
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 0
<
C0
D1
E2
F3"
trackable_list_wrapper
.
C0
D1"
trackable_list_wrapper
 "
trackable_list_wrapper
▓
├non_trainable_variables
─layers
┼metrics
 ╞layer_regularization_losses
╟layer_metrics
<	variables
=trainable_variables
>regularization_losses
@__call__
*A&call_and_return_all_conditional_losses
&A"call_and_return_conditional_losses"
_generic_user_object
╜
╚trace_0
╔trace_12В
#__inference_bn2_layer_call_fn_11566
#__inference_bn2_layer_call_fn_11579╡
о▓к
FullArgSpec)
args!Ъ
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsв
p 

 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 z╚trace_0z╔trace_1
є
╩trace_0
╦trace_12╕
>__inference_bn2_layer_call_and_return_conditional_losses_11597
>__inference_bn2_layer_call_and_return_conditional_losses_11615╡
о▓к
FullArgSpec)
args!Ъ
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsв
p 

 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 z╩trace_0z╦trace_1
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
▓
╠non_trainable_variables
═layers
╬metrics
 ╧layer_regularization_losses
╨layer_metrics
G	variables
Htrainable_variables
Iregularization_losses
K__call__
*L&call_and_return_all_conditional_losses
&L"call_and_return_conditional_losses"
_generic_user_object
т
╤trace_02├
&__inference_lReLU2_layer_call_fn_11620Ш
С▓Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 z╤trace_0
¤
╥trace_02▐
A__inference_lReLU2_layer_call_and_return_conditional_losses_11625Ш
С▓Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 z╥trace_0
.
S0
T1"
trackable_list_wrapper
.
S0
T1"
trackable_list_wrapper
 "
trackable_list_wrapper
▓
╙non_trainable_variables
╘layers
╒metrics
 ╓layer_regularization_losses
╫layer_metrics
M	variables
Ntrainable_variables
Oregularization_losses
Q__call__
*R&call_and_return_all_conditional_losses
&R"call_and_return_conditional_losses"
_generic_user_object
у
╪trace_02─
'__inference_conv2D3_layer_call_fn_11634Ш
С▓Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 z╪trace_0
■
┘trace_02▀
B__inference_conv2D3_layer_call_and_return_conditional_losses_11644Ш
С▓Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 z┘trace_0
(:&@@2conv2D3/kernel
:@2conv2D3/bias
к2зд
Ы▓Ч
FullArgSpec
argsЪ
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 0
<
]0
^1
_2
`3"
trackable_list_wrapper
.
]0
^1"
trackable_list_wrapper
 "
trackable_list_wrapper
▓
┌non_trainable_variables
█layers
▄metrics
 ▌layer_regularization_losses
▐layer_metrics
V	variables
Wtrainable_variables
Xregularization_losses
Z__call__
*[&call_and_return_all_conditional_losses
&["call_and_return_conditional_losses"
_generic_user_object
╜
▀trace_0
рtrace_12В
#__inference_bn3_layer_call_fn_11657
#__inference_bn3_layer_call_fn_11670╡
о▓к
FullArgSpec)
args!Ъ
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsв
p 

 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 z▀trace_0zрtrace_1
є
сtrace_0
тtrace_12╕
>__inference_bn3_layer_call_and_return_conditional_losses_11688
>__inference_bn3_layer_call_and_return_conditional_losses_11706╡
о▓к
FullArgSpec)
args!Ъ
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsв
p 

 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 zсtrace_0zтtrace_1
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
▓
уnon_trainable_variables
фlayers
хmetrics
 цlayer_regularization_losses
чlayer_metrics
a	variables
btrainable_variables
cregularization_losses
e__call__
*f&call_and_return_all_conditional_losses
&f"call_and_return_conditional_losses"
_generic_user_object
т
шtrace_02├
&__inference_lReLU3_layer_call_fn_11711Ш
С▓Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 zшtrace_0
¤
щtrace_02▐
A__inference_lReLU3_layer_call_and_return_conditional_losses_11716Ш
С▓Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 zщtrace_0
.
m0
n1"
trackable_list_wrapper
.
m0
n1"
trackable_list_wrapper
 "
trackable_list_wrapper
▓
ъnon_trainable_variables
ыlayers
ьmetrics
 эlayer_regularization_losses
юlayer_metrics
g	variables
htrainable_variables
iregularization_losses
k__call__
*l&call_and_return_all_conditional_losses
&l"call_and_return_conditional_losses"
_generic_user_object
у
яtrace_02─
'__inference_conv2D4_layer_call_fn_11725Ш
С▓Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 zяtrace_0
■
Ёtrace_02▀
B__inference_conv2D4_layer_call_and_return_conditional_losses_11735Ш
С▓Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 zЁtrace_0
(:&@@2conv2D4/kernel
:@2conv2D4/bias
к2зд
Ы▓Ч
FullArgSpec
argsЪ
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 0
<
w0
x1
y2
z3"
trackable_list_wrapper
.
w0
x1"
trackable_list_wrapper
 "
trackable_list_wrapper
▓
ёnon_trainable_variables
Єlayers
єmetrics
 Їlayer_regularization_losses
їlayer_metrics
p	variables
qtrainable_variables
rregularization_losses
t__call__
*u&call_and_return_all_conditional_losses
&u"call_and_return_conditional_losses"
_generic_user_object
╜
Ўtrace_0
ўtrace_12В
#__inference_bn4_layer_call_fn_11748
#__inference_bn4_layer_call_fn_11761╡
о▓к
FullArgSpec)
args!Ъ
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsв
p 

 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 zЎtrace_0zўtrace_1
є
°trace_0
∙trace_12╕
>__inference_bn4_layer_call_and_return_conditional_losses_11779
>__inference_bn4_layer_call_and_return_conditional_losses_11797╡
о▓к
FullArgSpec)
args!Ъ
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsв
p 

 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 z°trace_0z∙trace_1
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
┤
·non_trainable_variables
√layers
№metrics
 ¤layer_regularization_losses
■layer_metrics
{	variables
|trainable_variables
}regularization_losses
__call__
+А&call_and_return_all_conditional_losses
'А"call_and_return_conditional_losses"
_generic_user_object
т
 trace_02├
&__inference_lReLU4_layer_call_fn_11802Ш
С▓Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 z trace_0
¤
Аtrace_02▐
A__inference_lReLU4_layer_call_and_return_conditional_losses_11807Ш
С▓Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 zАtrace_0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
╕
Бnon_trainable_variables
Вlayers
Гmetrics
 Дlayer_regularization_losses
Еlayer_metrics
Б	variables
Вtrainable_variables
Гregularization_losses
Е__call__
+Ж&call_and_return_all_conditional_losses
'Ж"call_and_return_conditional_losses"
_generic_user_object
у
Жtrace_02─
'__inference_flatten_layer_call_fn_11812Ш
С▓Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 zЖtrace_0
■
Зtrace_02▀
B__inference_flatten_layer_call_and_return_conditional_losses_11818Ш
С▓Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 zЗtrace_0
0
Н0
О1"
trackable_list_wrapper
0
Н0
О1"
trackable_list_wrapper
 "
trackable_list_wrapper
╕
Иnon_trainable_variables
Йlayers
Кmetrics
 Лlayer_regularization_losses
Мlayer_metrics
З	variables
Иtrainable_variables
Йregularization_losses
Л__call__
+М&call_and_return_all_conditional_losses
'М"call_and_return_conditional_losses"
_generic_user_object
р
Нtrace_02┴
$__inference_mean_layer_call_fn_11827Ш
С▓Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 zНtrace_0
√
Оtrace_02▄
?__inference_mean_layer_call_and_return_conditional_losses_11837Ш
С▓Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 zОtrace_0
:	└
2mean/kernel
:
2	mean/bias
0
Х0
Ц1"
trackable_list_wrapper
0
Х0
Ц1"
trackable_list_wrapper
 "
trackable_list_wrapper
╕
Пnon_trainable_variables
Рlayers
Сmetrics
 Тlayer_regularization_losses
Уlayer_metrics
П	variables
Рtrainable_variables
Сregularization_losses
У__call__
+Ф&call_and_return_all_conditional_losses
'Ф"call_and_return_conditional_losses"
_generic_user_object
т
Фtrace_02├
&__inference_logVar_layer_call_fn_11846Ш
С▓Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 zФtrace_0
¤
Хtrace_02▐
A__inference_logVar_layer_call_and_return_conditional_losses_11856Ш
С▓Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 zХtrace_0
 :	└
2logVar/kernel
:
2logVar/bias
X
+0
,1
E2
F3
_4
`5
y6
z7"
trackable_list_wrapper
Ц
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
15"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
єBЁ
'__inference_encoder_layer_call_fn_10691input_layer"╡
о▓к
FullArgSpec)
args!Ъ
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsв
p 

 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
єBЁ
'__inference_encoder_layer_call_fn_10830input_layer"╡
о▓к
FullArgSpec)
args!Ъ
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsв
p 

 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
юBы
'__inference_encoder_layer_call_fn_11174inputs"╡
о▓к
FullArgSpec)
args!Ъ
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsв
p 

 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
юBы
'__inference_encoder_layer_call_fn_11237inputs"╡
о▓к
FullArgSpec)
args!Ъ
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsв
p 

 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
ОBЛ
B__inference_encoder_layer_call_and_return_conditional_losses_10475input_layer"╡
о▓к
FullArgSpec)
args!Ъ
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsв
p 

 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
ОBЛ
B__inference_encoder_layer_call_and_return_conditional_losses_10551input_layer"╡
о▓к
FullArgSpec)
args!Ъ
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsв
p 

 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
ЙBЖ
B__inference_encoder_layer_call_and_return_conditional_losses_11340inputs"╡
о▓к
FullArgSpec)
args!Ъ
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsв
p 

 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
ЙBЖ
B__inference_encoder_layer_call_and_return_conditional_losses_11443inputs"╡
о▓к
FullArgSpec)
args!Ъ
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsв
p 

 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
╬B╦
#__inference_signature_wrapper_11111input_layer"Ф
Н▓Й
FullArgSpec
argsЪ 
varargs
 
varkwjkwargs
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
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
╤B╬
'__inference_conv2D1_layer_call_fn_11452inputs"Ш
С▓Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
ьBщ
B__inference_conv2D1_layer_call_and_return_conditional_losses_11462inputs"Ш
С▓Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
.
+0
,1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
ъBч
#__inference_bn1_layer_call_fn_11475inputs"╡
о▓к
FullArgSpec)
args!Ъ
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsв
p 

 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
ъBч
#__inference_bn1_layer_call_fn_11488inputs"╡
о▓к
FullArgSpec)
args!Ъ
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsв
p 

 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
ЕBВ
>__inference_bn1_layer_call_and_return_conditional_losses_11506inputs"╡
о▓к
FullArgSpec)
args!Ъ
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsв
p 

 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
ЕBВ
>__inference_bn1_layer_call_and_return_conditional_losses_11524inputs"╡
о▓к
FullArgSpec)
args!Ъ
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsв
p 

 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
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
╨B═
&__inference_lReLU1_layer_call_fn_11529inputs"Ш
С▓Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
ыBш
A__inference_lReLU1_layer_call_and_return_conditional_losses_11534inputs"Ш
С▓Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
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
╤B╬
'__inference_conv2D2_layer_call_fn_11543inputs"Ш
С▓Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
ьBщ
B__inference_conv2D2_layer_call_and_return_conditional_losses_11553inputs"Ш
С▓Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
.
E0
F1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
ъBч
#__inference_bn2_layer_call_fn_11566inputs"╡
о▓к
FullArgSpec)
args!Ъ
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsв
p 

 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
ъBч
#__inference_bn2_layer_call_fn_11579inputs"╡
о▓к
FullArgSpec)
args!Ъ
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsв
p 

 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
ЕBВ
>__inference_bn2_layer_call_and_return_conditional_losses_11597inputs"╡
о▓к
FullArgSpec)
args!Ъ
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsв
p 

 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
ЕBВ
>__inference_bn2_layer_call_and_return_conditional_losses_11615inputs"╡
о▓к
FullArgSpec)
args!Ъ
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsв
p 

 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
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
╨B═
&__inference_lReLU2_layer_call_fn_11620inputs"Ш
С▓Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
ыBш
A__inference_lReLU2_layer_call_and_return_conditional_losses_11625inputs"Ш
С▓Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
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
╤B╬
'__inference_conv2D3_layer_call_fn_11634inputs"Ш
С▓Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
ьBщ
B__inference_conv2D3_layer_call_and_return_conditional_losses_11644inputs"Ш
С▓Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
.
_0
`1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
ъBч
#__inference_bn3_layer_call_fn_11657inputs"╡
о▓к
FullArgSpec)
args!Ъ
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsв
p 

 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
ъBч
#__inference_bn3_layer_call_fn_11670inputs"╡
о▓к
FullArgSpec)
args!Ъ
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsв
p 

 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
ЕBВ
>__inference_bn3_layer_call_and_return_conditional_losses_11688inputs"╡
о▓к
FullArgSpec)
args!Ъ
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsв
p 

 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
ЕBВ
>__inference_bn3_layer_call_and_return_conditional_losses_11706inputs"╡
о▓к
FullArgSpec)
args!Ъ
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsв
p 

 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
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
╨B═
&__inference_lReLU3_layer_call_fn_11711inputs"Ш
С▓Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
ыBш
A__inference_lReLU3_layer_call_and_return_conditional_losses_11716inputs"Ш
С▓Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
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
╤B╬
'__inference_conv2D4_layer_call_fn_11725inputs"Ш
С▓Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
ьBщ
B__inference_conv2D4_layer_call_and_return_conditional_losses_11735inputs"Ш
С▓Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
.
y0
z1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
ъBч
#__inference_bn4_layer_call_fn_11748inputs"╡
о▓к
FullArgSpec)
args!Ъ
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsв
p 

 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
ъBч
#__inference_bn4_layer_call_fn_11761inputs"╡
о▓к
FullArgSpec)
args!Ъ
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsв
p 

 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
ЕBВ
>__inference_bn4_layer_call_and_return_conditional_losses_11779inputs"╡
о▓к
FullArgSpec)
args!Ъ
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsв
p 

 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
ЕBВ
>__inference_bn4_layer_call_and_return_conditional_losses_11797inputs"╡
о▓к
FullArgSpec)
args!Ъ
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsв
p 

 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
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
╨B═
&__inference_lReLU4_layer_call_fn_11802inputs"Ш
С▓Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
ыBш
A__inference_lReLU4_layer_call_and_return_conditional_losses_11807inputs"Ш
С▓Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
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
╤B╬
'__inference_flatten_layer_call_fn_11812inputs"Ш
С▓Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
ьBщ
B__inference_flatten_layer_call_and_return_conditional_losses_11818inputs"Ш
С▓Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
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
╬B╦
$__inference_mean_layer_call_fn_11827inputs"Ш
С▓Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
щBц
?__inference_mean_layer_call_and_return_conditional_losses_11837inputs"Ш
С▓Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
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
╨B═
&__inference_logVar_layer_call_fn_11846inputs"Ш
С▓Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
ыBш
A__inference_logVar_layer_call_and_return_conditional_losses_11856inputs"Ш
С▓Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 ▐
 __inference__wrapped_model_10045╣  )*+,9:CDEFST]^_`mnwxyzХЦНО<в9
2в/
-К*
input_layer         
к "WкT
*
logVar К
logvar         

&
meanК
mean         
ф
>__inference_bn1_layer_call_and_return_conditional_losses_11506б)*+,QвN
GвD
:К7
inputs+                            
p

 
к "FвC
<К9
tensor_0+                            
Ъ ф
>__inference_bn1_layer_call_and_return_conditional_losses_11524б)*+,QвN
GвD
:К7
inputs+                            
p 

 
к "FвC
<К9
tensor_0+                            
Ъ ╛
#__inference_bn1_layer_call_fn_11475Ц)*+,QвN
GвD
:К7
inputs+                            
p

 
к ";К8
unknown+                            ╛
#__inference_bn1_layer_call_fn_11488Ц)*+,QвN
GвD
:К7
inputs+                            
p 

 
к ";К8
unknown+                            ф
>__inference_bn2_layer_call_and_return_conditional_losses_11597бCDEFQвN
GвD
:К7
inputs+                           @
p

 
к "FвC
<К9
tensor_0+                           @
Ъ ф
>__inference_bn2_layer_call_and_return_conditional_losses_11615бCDEFQвN
GвD
:К7
inputs+                           @
p 

 
к "FвC
<К9
tensor_0+                           @
Ъ ╛
#__inference_bn2_layer_call_fn_11566ЦCDEFQвN
GвD
:К7
inputs+                           @
p

 
к ";К8
unknown+                           @╛
#__inference_bn2_layer_call_fn_11579ЦCDEFQвN
GвD
:К7
inputs+                           @
p 

 
к ";К8
unknown+                           @ф
>__inference_bn3_layer_call_and_return_conditional_losses_11688б]^_`QвN
GвD
:К7
inputs+                           @
p

 
к "FвC
<К9
tensor_0+                           @
Ъ ф
>__inference_bn3_layer_call_and_return_conditional_losses_11706б]^_`QвN
GвD
:К7
inputs+                           @
p 

 
к "FвC
<К9
tensor_0+                           @
Ъ ╛
#__inference_bn3_layer_call_fn_11657Ц]^_`QвN
GвD
:К7
inputs+                           @
p

 
к ";К8
unknown+                           @╛
#__inference_bn3_layer_call_fn_11670Ц]^_`QвN
GвD
:К7
inputs+                           @
p 

 
к ";К8
unknown+                           @ф
>__inference_bn4_layer_call_and_return_conditional_losses_11779бwxyzQвN
GвD
:К7
inputs+                           @
p

 
к "FвC
<К9
tensor_0+                           @
Ъ ф
>__inference_bn4_layer_call_and_return_conditional_losses_11797бwxyzQвN
GвD
:К7
inputs+                           @
p 

 
к "FвC
<К9
tensor_0+                           @
Ъ ╛
#__inference_bn4_layer_call_fn_11748ЦwxyzQвN
GвD
:К7
inputs+                           @
p

 
к ";К8
unknown+                           @╛
#__inference_bn4_layer_call_fn_11761ЦwxyzQвN
GвD
:К7
inputs+                           @
p 

 
к ";К8
unknown+                           @╣
B__inference_conv2D1_layer_call_and_return_conditional_losses_11462s 7в4
-в*
(К%
inputs         
к "4в1
*К'
tensor_0          
Ъ У
'__inference_conv2D1_layer_call_fn_11452h 7в4
-в*
(К%
inputs         
к ")К&
unknown          ╣
B__inference_conv2D2_layer_call_and_return_conditional_losses_11553s9:7в4
-в*
(К%
inputs          
к "4в1
*К'
tensor_0         @
Ъ У
'__inference_conv2D2_layer_call_fn_11543h9:7в4
-в*
(К%
inputs          
к ")К&
unknown         @╣
B__inference_conv2D3_layer_call_and_return_conditional_losses_11644sST7в4
-в*
(К%
inputs         @
к "4в1
*К'
tensor_0         @
Ъ У
'__inference_conv2D3_layer_call_fn_11634hST7в4
-в*
(К%
inputs         @
к ")К&
unknown         @╣
B__inference_conv2D4_layer_call_and_return_conditional_losses_11735smn7в4
-в*
(К%
inputs         @
к "4в1
*К'
tensor_0         @
Ъ У
'__inference_conv2D4_layer_call_fn_11725hmn7в4
-в*
(К%
inputs         @
к ")К&
unknown         @К
B__inference_encoder_layer_call_and_return_conditional_losses_10475├  )*+,9:CDEFST]^_`mnwxyzХЦНОDвA
:в7
-К*
input_layer         
p

 
к "YвV
OвL
$К!

tensor_0_0         

$К!

tensor_0_1         

Ъ К
B__inference_encoder_layer_call_and_return_conditional_losses_10551├  )*+,9:CDEFST]^_`mnwxyzХЦНОDвA
:в7
-К*
input_layer         
p 

 
к "YвV
OвL
$К!

tensor_0_0         

$К!

tensor_0_1         

Ъ Е
B__inference_encoder_layer_call_and_return_conditional_losses_11340╛  )*+,9:CDEFST]^_`mnwxyzХЦНО?в<
5в2
(К%
inputs         
p

 
к "YвV
OвL
$К!

tensor_0_0         

$К!

tensor_0_1         

Ъ Е
B__inference_encoder_layer_call_and_return_conditional_losses_11443╛  )*+,9:CDEFST]^_`mnwxyzХЦНО?в<
5в2
(К%
inputs         
p 

 
к "YвV
OвL
$К!

tensor_0_0         

$К!

tensor_0_1         

Ъ с
'__inference_encoder_layer_call_fn_10691╡  )*+,9:CDEFST]^_`mnwxyzХЦНОDвA
:в7
-К*
input_layer         
p

 
к "KвH
"К
tensor_0         

"К
tensor_1         
с
'__inference_encoder_layer_call_fn_10830╡  )*+,9:CDEFST]^_`mnwxyzХЦНОDвA
:в7
-К*
input_layer         
p 

 
к "KвH
"К
tensor_0         

"К
tensor_1         
▄
'__inference_encoder_layer_call_fn_11174░  )*+,9:CDEFST]^_`mnwxyzХЦНО?в<
5в2
(К%
inputs         
p

 
к "KвH
"К
tensor_0         

"К
tensor_1         
▄
'__inference_encoder_layer_call_fn_11237░  )*+,9:CDEFST]^_`mnwxyzХЦНО?в<
5в2
(К%
inputs         
p 

 
к "KвH
"К
tensor_0         

"К
tensor_1         
о
B__inference_flatten_layer_call_and_return_conditional_losses_11818h7в4
-в*
(К%
inputs         @
к "-в*
#К 
tensor_0         └
Ъ И
'__inference_flatten_layer_call_fn_11812]7в4
-в*
(К%
inputs         @
к ""К
unknown         └┤
A__inference_lReLU1_layer_call_and_return_conditional_losses_11534o7в4
-в*
(К%
inputs          
к "4в1
*К'
tensor_0          
Ъ О
&__inference_lReLU1_layer_call_fn_11529d7в4
-в*
(К%
inputs          
к ")К&
unknown          ┤
A__inference_lReLU2_layer_call_and_return_conditional_losses_11625o7в4
-в*
(К%
inputs         @
к "4в1
*К'
tensor_0         @
Ъ О
&__inference_lReLU2_layer_call_fn_11620d7в4
-в*
(К%
inputs         @
к ")К&
unknown         @┤
A__inference_lReLU3_layer_call_and_return_conditional_losses_11716o7в4
-в*
(К%
inputs         @
к "4в1
*К'
tensor_0         @
Ъ О
&__inference_lReLU3_layer_call_fn_11711d7в4
-в*
(К%
inputs         @
к ")К&
unknown         @┤
A__inference_lReLU4_layer_call_and_return_conditional_losses_11807o7в4
-в*
(К%
inputs         @
к "4в1
*К'
tensor_0         @
Ъ О
&__inference_lReLU4_layer_call_fn_11802d7в4
-в*
(К%
inputs         @
к ")К&
unknown         @л
A__inference_logVar_layer_call_and_return_conditional_losses_11856fХЦ0в-
&в#
!К
inputs         └
к ",в)
"К
tensor_0         

Ъ Е
&__inference_logVar_layer_call_fn_11846[ХЦ0в-
&в#
!К
inputs         └
к "!К
unknown         
й
?__inference_mean_layer_call_and_return_conditional_losses_11837fНО0в-
&в#
!К
inputs         └
к ",в)
"К
tensor_0         

Ъ Г
$__inference_mean_layer_call_fn_11827[НО0в-
&в#
!К
inputs         └
к "!К
unknown         
Ё
#__inference_signature_wrapper_11111╚  )*+,9:CDEFST]^_`mnwxyzХЦНОKвH
в 
Aк>
<
input_layer-К*
input_layer         "WкT
*
logVar К
logvar         

&
meanК
mean         
