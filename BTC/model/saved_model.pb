¬³
³
B
AddV2
x"T
y"T
z"T"
Ttype:
2	
B
AssignVariableOp
resource
value"dtype"
dtypetype
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
8
Const
output"dtype"
valuetensor"
dtypetype
^
Fill
dims"
index_type

value"T
output"T"	
Ttype"

index_typetype0:
2	
.
Identity

input"T
output"T"	
Ttype
:
Less
x"T
y"T
z
"
Ttype:
2	
q
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:

2	
e
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool(
=
Mul
x"T
y"T
z"T"
Ttype:
2	
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
dtypetype
E
Relu
features"T
activations"T"
Ttype:
2	
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0
?
Select
	condition

t"T
e"T
output"T"	
Ttype
P
Shape

input"T
output"out_type"	
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
[
Split
	split_dim

value"T
output"T*	num_split"
	num_splitint(0"	
Ttype
¾
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
executor_typestring 
@
StaticRegexFullMatch	
input

output
"
patternstring
ö
StridedSlice

input"T
begin"Index
end"Index
strides"Index
output"T"	
Ttype"
Indextype:
2	"

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

TensorListFromTensor
tensor"element_dtype
element_shape"
shape_type
output_handle"
element_dtypetype"

shape_typetype:
2	

TensorListReserve
element_shape"
shape_type
num_elements

handle"
element_dtypetype"

shape_typetype:
2	

TensorListStack
input_handle
element_shape
tensor"element_dtype"
element_dtypetype" 
num_elementsintÿÿÿÿÿÿÿÿÿ
P
	Transpose
x"T
perm"Tperm
y"T"	
Ttype"
Tpermtype0:
2	

VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 

While

input2T
output2T"
T
list(type)("
condfunc"
bodyfunc" 
output_shapeslist(shape)
 "
parallel_iterationsint
"serve*2.4.12v2.4.1-0-g85c8b2a817f8úû
z
dense_71/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:d* 
shared_namedense_71/kernel
s
#dense_71/kernel/Read/ReadVariableOpReadVariableOpdense_71/kernel*
_output_shapes

:d*
dtype0
r
dense_71/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_71/bias
k
!dense_71/bias/Read/ReadVariableOpReadVariableOpdense_71/bias*
_output_shapes
:*
dtype0
f
	Adam/iterVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_name	Adam/iter
_
Adam/iter/Read/ReadVariableOpReadVariableOp	Adam/iter*
_output_shapes
: *
dtype0	
j
Adam/beta_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_1
c
Adam/beta_1/Read/ReadVariableOpReadVariableOpAdam/beta_1*
_output_shapes
: *
dtype0
j
Adam/beta_2VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_2
c
Adam/beta_2/Read/ReadVariableOpReadVariableOpAdam/beta_2*
_output_shapes
: *
dtype0
h

Adam/decayVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name
Adam/decay
a
Adam/decay/Read/ReadVariableOpReadVariableOp
Adam/decay*
_output_shapes
: *
dtype0
x
Adam/learning_rateVarHandleOp*
_output_shapes
: *
dtype0*
shape: *#
shared_nameAdam/learning_rate
q
&Adam/learning_rate/Read/ReadVariableOpReadVariableOpAdam/learning_rate*
_output_shapes
: *
dtype0

lstm_71/lstm_cell_71/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	*,
shared_namelstm_71/lstm_cell_71/kernel

/lstm_71/lstm_cell_71/kernel/Read/ReadVariableOpReadVariableOplstm_71/lstm_cell_71/kernel*
_output_shapes
:	*
dtype0
§
%lstm_71/lstm_cell_71/recurrent_kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	d*6
shared_name'%lstm_71/lstm_cell_71/recurrent_kernel
 
9lstm_71/lstm_cell_71/recurrent_kernel/Read/ReadVariableOpReadVariableOp%lstm_71/lstm_cell_71/recurrent_kernel*
_output_shapes
:	d*
dtype0

lstm_71/lstm_cell_71/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:**
shared_namelstm_71/lstm_cell_71/bias

-lstm_71/lstm_cell_71/bias/Read/ReadVariableOpReadVariableOplstm_71/lstm_cell_71/bias*
_output_shapes	
:*
dtype0
^
totalVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nametotal
W
total/Read/ReadVariableOpReadVariableOptotal*
_output_shapes
: *
dtype0
^
countVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namecount
W
count/Read/ReadVariableOpReadVariableOpcount*
_output_shapes
: *
dtype0

Adam/dense_71/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:d*'
shared_nameAdam/dense_71/kernel/m

*Adam/dense_71/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_71/kernel/m*
_output_shapes

:d*
dtype0

Adam/dense_71/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/dense_71/bias/m
y
(Adam/dense_71/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_71/bias/m*
_output_shapes
:*
dtype0
¡
"Adam/lstm_71/lstm_cell_71/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	*3
shared_name$"Adam/lstm_71/lstm_cell_71/kernel/m

6Adam/lstm_71/lstm_cell_71/kernel/m/Read/ReadVariableOpReadVariableOp"Adam/lstm_71/lstm_cell_71/kernel/m*
_output_shapes
:	*
dtype0
µ
,Adam/lstm_71/lstm_cell_71/recurrent_kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	d*=
shared_name.,Adam/lstm_71/lstm_cell_71/recurrent_kernel/m
®
@Adam/lstm_71/lstm_cell_71/recurrent_kernel/m/Read/ReadVariableOpReadVariableOp,Adam/lstm_71/lstm_cell_71/recurrent_kernel/m*
_output_shapes
:	d*
dtype0

 Adam/lstm_71/lstm_cell_71/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*1
shared_name" Adam/lstm_71/lstm_cell_71/bias/m

4Adam/lstm_71/lstm_cell_71/bias/m/Read/ReadVariableOpReadVariableOp Adam/lstm_71/lstm_cell_71/bias/m*
_output_shapes	
:*
dtype0

Adam/dense_71/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:d*'
shared_nameAdam/dense_71/kernel/v

*Adam/dense_71/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_71/kernel/v*
_output_shapes

:d*
dtype0

Adam/dense_71/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/dense_71/bias/v
y
(Adam/dense_71/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_71/bias/v*
_output_shapes
:*
dtype0
¡
"Adam/lstm_71/lstm_cell_71/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	*3
shared_name$"Adam/lstm_71/lstm_cell_71/kernel/v

6Adam/lstm_71/lstm_cell_71/kernel/v/Read/ReadVariableOpReadVariableOp"Adam/lstm_71/lstm_cell_71/kernel/v*
_output_shapes
:	*
dtype0
µ
,Adam/lstm_71/lstm_cell_71/recurrent_kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	d*=
shared_name.,Adam/lstm_71/lstm_cell_71/recurrent_kernel/v
®
@Adam/lstm_71/lstm_cell_71/recurrent_kernel/v/Read/ReadVariableOpReadVariableOp,Adam/lstm_71/lstm_cell_71/recurrent_kernel/v*
_output_shapes
:	d*
dtype0

 Adam/lstm_71/lstm_cell_71/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*1
shared_name" Adam/lstm_71/lstm_cell_71/bias/v

4Adam/lstm_71/lstm_cell_71/bias/v/Read/ReadVariableOpReadVariableOp Adam/lstm_71/lstm_cell_71/bias/v*
_output_shapes	
:*
dtype0

NoOpNoOp
¸ 
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*ó
valueéBæ Bß
¿
layer_with_weights-0
layer-0
layer_with_weights-1
layer-1
	optimizer
	variables
regularization_losses
trainable_variables
	keras_api

signatures
l
	cell


state_spec
	variables
regularization_losses
trainable_variables
	keras_api
h

kernel
bias
	variables
regularization_losses
trainable_variables
	keras_api

iter

beta_1

beta_2
	decay
learning_ratem;m<m=m>m?v@vAvBvCvD
#
0
1
2
3
4
 
#
0
1
2
3
4
­
metrics
	variables
layer_regularization_losses
layer_metrics

 layers
!non_trainable_variables
regularization_losses
trainable_variables
 
~

kernel
recurrent_kernel
bias
"	variables
#regularization_losses
$trainable_variables
%	keras_api
 

0
1
2
 

0
1
2
¹
&metrics
	variables
'layer_regularization_losses
(layer_metrics

)states

*layers
+non_trainable_variables
regularization_losses
trainable_variables
[Y
VARIABLE_VALUEdense_71/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEdense_71/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE

0
1
 

0
1
­
,metrics
	variables
-layer_regularization_losses
.layer_metrics

/layers
0non_trainable_variables
regularization_losses
trainable_variables
HF
VARIABLE_VALUE	Adam/iter)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEAdam/beta_1+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEAdam/beta_2+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUE
JH
VARIABLE_VALUE
Adam/decay*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUE
ZX
VARIABLE_VALUEAdam/learning_rate2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUElstm_71/lstm_cell_71/kernel&variables/0/.ATTRIBUTES/VARIABLE_VALUE
a_
VARIABLE_VALUE%lstm_71/lstm_cell_71/recurrent_kernel&variables/1/.ATTRIBUTES/VARIABLE_VALUE
US
VARIABLE_VALUElstm_71/lstm_cell_71/bias&variables/2/.ATTRIBUTES/VARIABLE_VALUE

10
 
 

0
1
 

0
1
2
 

0
1
2
­
2metrics
"	variables
3layer_regularization_losses
4layer_metrics

5layers
6non_trainable_variables
#regularization_losses
$trainable_variables
 
 
 
 

	0
 
 
 
 
 
 
4
	7total
	8count
9	variables
:	keras_api
 
 
 
 
 
OM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE
OM
VARIABLE_VALUEcount4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE

70
81

9	variables
~|
VARIABLE_VALUEAdam/dense_71/kernel/mRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/dense_71/bias/mPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUE"Adam/lstm_71/lstm_cell_71/kernel/mBvariables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE,Adam/lstm_71/lstm_cell_71/recurrent_kernel/mBvariables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
xv
VARIABLE_VALUE Adam/lstm_71/lstm_cell_71/bias/mBvariables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/dense_71/kernel/vRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/dense_71/bias/vPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUE"Adam/lstm_71/lstm_cell_71/kernel/vBvariables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE,Adam/lstm_71/lstm_cell_71/recurrent_kernel/vBvariables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
xv
VARIABLE_VALUE Adam/lstm_71/lstm_cell_71/bias/vBvariables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE

serving_default_lstm_71_inputPlaceholder*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype0* 
shape:ÿÿÿÿÿÿÿÿÿ
Á
StatefulPartitionedCallStatefulPartitionedCallserving_default_lstm_71_inputlstm_71/lstm_cell_71/kernel%lstm_71/lstm_cell_71/recurrent_kernellstm_71/lstm_cell_71/biasdense_71/kerneldense_71/bias*
Tin

2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*'
_read_only_resource_inputs	
*-
config_proto

CPU

GPU 2J 8 *-
f(R&
$__inference_signature_wrapper_624375
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
ÿ	
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename#dense_71/kernel/Read/ReadVariableOp!dense_71/bias/Read/ReadVariableOpAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOp&Adam/learning_rate/Read/ReadVariableOp/lstm_71/lstm_cell_71/kernel/Read/ReadVariableOp9lstm_71/lstm_cell_71/recurrent_kernel/Read/ReadVariableOp-lstm_71/lstm_cell_71/bias/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOp*Adam/dense_71/kernel/m/Read/ReadVariableOp(Adam/dense_71/bias/m/Read/ReadVariableOp6Adam/lstm_71/lstm_cell_71/kernel/m/Read/ReadVariableOp@Adam/lstm_71/lstm_cell_71/recurrent_kernel/m/Read/ReadVariableOp4Adam/lstm_71/lstm_cell_71/bias/m/Read/ReadVariableOp*Adam/dense_71/kernel/v/Read/ReadVariableOp(Adam/dense_71/bias/v/Read/ReadVariableOp6Adam/lstm_71/lstm_cell_71/kernel/v/Read/ReadVariableOp@Adam/lstm_71/lstm_cell_71/recurrent_kernel/v/Read/ReadVariableOp4Adam/lstm_71/lstm_cell_71/bias/v/Read/ReadVariableOpConst*#
Tin
2	*
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
GPU 2J 8 *(
f#R!
__inference__traced_save_625587
Â
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamedense_71/kerneldense_71/bias	Adam/iterAdam/beta_1Adam/beta_2
Adam/decayAdam/learning_ratelstm_71/lstm_cell_71/kernel%lstm_71/lstm_cell_71/recurrent_kernellstm_71/lstm_cell_71/biastotalcountAdam/dense_71/kernel/mAdam/dense_71/bias/m"Adam/lstm_71/lstm_cell_71/kernel/m,Adam/lstm_71/lstm_cell_71/recurrent_kernel/m Adam/lstm_71/lstm_cell_71/bias/mAdam/dense_71/kernel/vAdam/dense_71/bias/v"Adam/lstm_71/lstm_cell_71/kernel/v,Adam/lstm_71/lstm_cell_71/recurrent_kernel/v Adam/lstm_71/lstm_cell_71/bias/v*"
Tin
2*
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
GPU 2J 8 *+
f&R$
"__inference__traced_restore_625663ÿ
þc
À
'sequential_71_lstm_71_while_body_623204H
Dsequential_71_lstm_71_while_sequential_71_lstm_71_while_loop_counterN
Jsequential_71_lstm_71_while_sequential_71_lstm_71_while_maximum_iterations+
'sequential_71_lstm_71_while_placeholder-
)sequential_71_lstm_71_while_placeholder_1-
)sequential_71_lstm_71_while_placeholder_2-
)sequential_71_lstm_71_while_placeholder_3G
Csequential_71_lstm_71_while_sequential_71_lstm_71_strided_slice_1_0
sequential_71_lstm_71_while_tensorarrayv2read_tensorlistgetitem_sequential_71_lstm_71_tensorarrayunstack_tensorlistfromtensor_0M
Isequential_71_lstm_71_while_lstm_cell_71_matmul_readvariableop_resource_0O
Ksequential_71_lstm_71_while_lstm_cell_71_matmul_1_readvariableop_resource_0N
Jsequential_71_lstm_71_while_lstm_cell_71_biasadd_readvariableop_resource_0(
$sequential_71_lstm_71_while_identity*
&sequential_71_lstm_71_while_identity_1*
&sequential_71_lstm_71_while_identity_2*
&sequential_71_lstm_71_while_identity_3*
&sequential_71_lstm_71_while_identity_4*
&sequential_71_lstm_71_while_identity_5E
Asequential_71_lstm_71_while_sequential_71_lstm_71_strided_slice_1
}sequential_71_lstm_71_while_tensorarrayv2read_tensorlistgetitem_sequential_71_lstm_71_tensorarrayunstack_tensorlistfromtensorK
Gsequential_71_lstm_71_while_lstm_cell_71_matmul_readvariableop_resourceM
Isequential_71_lstm_71_while_lstm_cell_71_matmul_1_readvariableop_resourceL
Hsequential_71_lstm_71_while_lstm_cell_71_biasadd_readvariableop_resource¢?sequential_71/lstm_71/while/lstm_cell_71/BiasAdd/ReadVariableOp¢>sequential_71/lstm_71/while/lstm_cell_71/MatMul/ReadVariableOp¢@sequential_71/lstm_71/while/lstm_cell_71/MatMul_1/ReadVariableOpï
Msequential_71/lstm_71/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   2O
Msequential_71/lstm_71/while/TensorArrayV2Read/TensorListGetItem/element_shape×
?sequential_71/lstm_71/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemsequential_71_lstm_71_while_tensorarrayv2read_tensorlistgetitem_sequential_71_lstm_71_tensorarrayunstack_tensorlistfromtensor_0'sequential_71_lstm_71_while_placeholderVsequential_71/lstm_71/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
element_dtype02A
?sequential_71/lstm_71/while/TensorArrayV2Read/TensorListGetItem
>sequential_71/lstm_71/while/lstm_cell_71/MatMul/ReadVariableOpReadVariableOpIsequential_71_lstm_71_while_lstm_cell_71_matmul_readvariableop_resource_0*
_output_shapes
:	*
dtype02@
>sequential_71/lstm_71/while/lstm_cell_71/MatMul/ReadVariableOp¯
/sequential_71/lstm_71/while/lstm_cell_71/MatMulMatMulFsequential_71/lstm_71/while/TensorArrayV2Read/TensorListGetItem:item:0Fsequential_71/lstm_71/while/lstm_cell_71/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ21
/sequential_71/lstm_71/while/lstm_cell_71/MatMul
@sequential_71/lstm_71/while/lstm_cell_71/MatMul_1/ReadVariableOpReadVariableOpKsequential_71_lstm_71_while_lstm_cell_71_matmul_1_readvariableop_resource_0*
_output_shapes
:	d*
dtype02B
@sequential_71/lstm_71/while/lstm_cell_71/MatMul_1/ReadVariableOp
1sequential_71/lstm_71/while/lstm_cell_71/MatMul_1MatMul)sequential_71_lstm_71_while_placeholder_2Hsequential_71/lstm_71/while/lstm_cell_71/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ23
1sequential_71/lstm_71/while/lstm_cell_71/MatMul_1
,sequential_71/lstm_71/while/lstm_cell_71/addAddV29sequential_71/lstm_71/while/lstm_cell_71/MatMul:product:0;sequential_71/lstm_71/while/lstm_cell_71/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2.
,sequential_71/lstm_71/while/lstm_cell_71/add
?sequential_71/lstm_71/while/lstm_cell_71/BiasAdd/ReadVariableOpReadVariableOpJsequential_71_lstm_71_while_lstm_cell_71_biasadd_readvariableop_resource_0*
_output_shapes	
:*
dtype02A
?sequential_71/lstm_71/while/lstm_cell_71/BiasAdd/ReadVariableOp
0sequential_71/lstm_71/while/lstm_cell_71/BiasAddBiasAdd0sequential_71/lstm_71/while/lstm_cell_71/add:z:0Gsequential_71/lstm_71/while/lstm_cell_71/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22
0sequential_71/lstm_71/while/lstm_cell_71/BiasAdd¢
.sequential_71/lstm_71/while/lstm_cell_71/ConstConst*
_output_shapes
: *
dtype0*
value	B :20
.sequential_71/lstm_71/while/lstm_cell_71/Const¶
8sequential_71/lstm_71/while/lstm_cell_71/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2:
8sequential_71/lstm_71/while/lstm_cell_71/split/split_dimã
.sequential_71/lstm_71/while/lstm_cell_71/splitSplitAsequential_71/lstm_71/while/lstm_cell_71/split/split_dim:output:09sequential_71/lstm_71/while/lstm_cell_71/BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd*
	num_split20
.sequential_71/lstm_71/while/lstm_cell_71/splitÚ
0sequential_71/lstm_71/while/lstm_cell_71/SigmoidSigmoid7sequential_71/lstm_71/while/lstm_cell_71/split:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd22
0sequential_71/lstm_71/while/lstm_cell_71/SigmoidÞ
2sequential_71/lstm_71/while/lstm_cell_71/Sigmoid_1Sigmoid7sequential_71/lstm_71/while/lstm_cell_71/split:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd24
2sequential_71/lstm_71/while/lstm_cell_71/Sigmoid_1ø
,sequential_71/lstm_71/while/lstm_cell_71/mulMul6sequential_71/lstm_71/while/lstm_cell_71/Sigmoid_1:y:0)sequential_71_lstm_71_while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd2.
,sequential_71/lstm_71/while/lstm_cell_71/mulÑ
-sequential_71/lstm_71/while/lstm_cell_71/ReluRelu7sequential_71/lstm_71/while/lstm_cell_71/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd2/
-sequential_71/lstm_71/while/lstm_cell_71/Relu
.sequential_71/lstm_71/while/lstm_cell_71/mul_1Mul4sequential_71/lstm_71/while/lstm_cell_71/Sigmoid:y:0;sequential_71/lstm_71/while/lstm_cell_71/Relu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd20
.sequential_71/lstm_71/while/lstm_cell_71/mul_1
.sequential_71/lstm_71/while/lstm_cell_71/add_1AddV20sequential_71/lstm_71/while/lstm_cell_71/mul:z:02sequential_71/lstm_71/while/lstm_cell_71/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd20
.sequential_71/lstm_71/while/lstm_cell_71/add_1Þ
2sequential_71/lstm_71/while/lstm_cell_71/Sigmoid_2Sigmoid7sequential_71/lstm_71/while/lstm_cell_71/split:output:3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd24
2sequential_71/lstm_71/while/lstm_cell_71/Sigmoid_2Ð
/sequential_71/lstm_71/while/lstm_cell_71/Relu_1Relu2sequential_71/lstm_71/while/lstm_cell_71/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd21
/sequential_71/lstm_71/while/lstm_cell_71/Relu_1
.sequential_71/lstm_71/while/lstm_cell_71/mul_2Mul6sequential_71/lstm_71/while/lstm_cell_71/Sigmoid_2:y:0=sequential_71/lstm_71/while/lstm_cell_71/Relu_1:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd20
.sequential_71/lstm_71/while/lstm_cell_71/mul_2Î
@sequential_71/lstm_71/while/TensorArrayV2Write/TensorListSetItemTensorListSetItem)sequential_71_lstm_71_while_placeholder_1'sequential_71_lstm_71_while_placeholder2sequential_71/lstm_71/while/lstm_cell_71/mul_2:z:0*
_output_shapes
: *
element_dtype02B
@sequential_71/lstm_71/while/TensorArrayV2Write/TensorListSetItem
!sequential_71/lstm_71/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2#
!sequential_71/lstm_71/while/add/yÁ
sequential_71/lstm_71/while/addAddV2'sequential_71_lstm_71_while_placeholder*sequential_71/lstm_71/while/add/y:output:0*
T0*
_output_shapes
: 2!
sequential_71/lstm_71/while/add
#sequential_71/lstm_71/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2%
#sequential_71/lstm_71/while/add_1/yä
!sequential_71/lstm_71/while/add_1AddV2Dsequential_71_lstm_71_while_sequential_71_lstm_71_while_loop_counter,sequential_71/lstm_71/while/add_1/y:output:0*
T0*
_output_shapes
: 2#
!sequential_71/lstm_71/while/add_1æ
$sequential_71/lstm_71/while/IdentityIdentity%sequential_71/lstm_71/while/add_1:z:0@^sequential_71/lstm_71/while/lstm_cell_71/BiasAdd/ReadVariableOp?^sequential_71/lstm_71/while/lstm_cell_71/MatMul/ReadVariableOpA^sequential_71/lstm_71/while/lstm_cell_71/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2&
$sequential_71/lstm_71/while/Identity
&sequential_71/lstm_71/while/Identity_1IdentityJsequential_71_lstm_71_while_sequential_71_lstm_71_while_maximum_iterations@^sequential_71/lstm_71/while/lstm_cell_71/BiasAdd/ReadVariableOp?^sequential_71/lstm_71/while/lstm_cell_71/MatMul/ReadVariableOpA^sequential_71/lstm_71/while/lstm_cell_71/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2(
&sequential_71/lstm_71/while/Identity_1è
&sequential_71/lstm_71/while/Identity_2Identity#sequential_71/lstm_71/while/add:z:0@^sequential_71/lstm_71/while/lstm_cell_71/BiasAdd/ReadVariableOp?^sequential_71/lstm_71/while/lstm_cell_71/MatMul/ReadVariableOpA^sequential_71/lstm_71/while/lstm_cell_71/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2(
&sequential_71/lstm_71/while/Identity_2
&sequential_71/lstm_71/while/Identity_3IdentityPsequential_71/lstm_71/while/TensorArrayV2Write/TensorListSetItem:output_handle:0@^sequential_71/lstm_71/while/lstm_cell_71/BiasAdd/ReadVariableOp?^sequential_71/lstm_71/while/lstm_cell_71/MatMul/ReadVariableOpA^sequential_71/lstm_71/while/lstm_cell_71/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2(
&sequential_71/lstm_71/while/Identity_3
&sequential_71/lstm_71/while/Identity_4Identity2sequential_71/lstm_71/while/lstm_cell_71/mul_2:z:0@^sequential_71/lstm_71/while/lstm_cell_71/BiasAdd/ReadVariableOp?^sequential_71/lstm_71/while/lstm_cell_71/MatMul/ReadVariableOpA^sequential_71/lstm_71/while/lstm_cell_71/MatMul_1/ReadVariableOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd2(
&sequential_71/lstm_71/while/Identity_4
&sequential_71/lstm_71/while/Identity_5Identity2sequential_71/lstm_71/while/lstm_cell_71/add_1:z:0@^sequential_71/lstm_71/while/lstm_cell_71/BiasAdd/ReadVariableOp?^sequential_71/lstm_71/while/lstm_cell_71/MatMul/ReadVariableOpA^sequential_71/lstm_71/while/lstm_cell_71/MatMul_1/ReadVariableOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd2(
&sequential_71/lstm_71/while/Identity_5"U
$sequential_71_lstm_71_while_identity-sequential_71/lstm_71/while/Identity:output:0"Y
&sequential_71_lstm_71_while_identity_1/sequential_71/lstm_71/while/Identity_1:output:0"Y
&sequential_71_lstm_71_while_identity_2/sequential_71/lstm_71/while/Identity_2:output:0"Y
&sequential_71_lstm_71_while_identity_3/sequential_71/lstm_71/while/Identity_3:output:0"Y
&sequential_71_lstm_71_while_identity_4/sequential_71/lstm_71/while/Identity_4:output:0"Y
&sequential_71_lstm_71_while_identity_5/sequential_71/lstm_71/while/Identity_5:output:0"
Hsequential_71_lstm_71_while_lstm_cell_71_biasadd_readvariableop_resourceJsequential_71_lstm_71_while_lstm_cell_71_biasadd_readvariableop_resource_0"
Isequential_71_lstm_71_while_lstm_cell_71_matmul_1_readvariableop_resourceKsequential_71_lstm_71_while_lstm_cell_71_matmul_1_readvariableop_resource_0"
Gsequential_71_lstm_71_while_lstm_cell_71_matmul_readvariableop_resourceIsequential_71_lstm_71_while_lstm_cell_71_matmul_readvariableop_resource_0"
Asequential_71_lstm_71_while_sequential_71_lstm_71_strided_slice_1Csequential_71_lstm_71_while_sequential_71_lstm_71_strided_slice_1_0"
}sequential_71_lstm_71_while_tensorarrayv2read_tensorlistgetitem_sequential_71_lstm_71_tensorarrayunstack_tensorlistfromtensorsequential_71_lstm_71_while_tensorarrayv2read_tensorlistgetitem_sequential_71_lstm_71_tensorarrayunstack_tensorlistfromtensor_0*Q
_input_shapes@
>: : : : :ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd: : :::2
?sequential_71/lstm_71/while/lstm_cell_71/BiasAdd/ReadVariableOp?sequential_71/lstm_71/while/lstm_cell_71/BiasAdd/ReadVariableOp2
>sequential_71/lstm_71/while/lstm_cell_71/MatMul/ReadVariableOp>sequential_71/lstm_71/while/lstm_cell_71/MatMul/ReadVariableOp2
@sequential_71/lstm_71/while/lstm_cell_71/MatMul_1/ReadVariableOp@sequential_71/lstm_71/while/lstm_cell_71/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd:-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd:

_output_shapes
: :

_output_shapes
: 
C
þ
while_body_623976
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_07
3while_lstm_cell_71_matmul_readvariableop_resource_09
5while_lstm_cell_71_matmul_1_readvariableop_resource_08
4while_lstm_cell_71_biasadd_readvariableop_resource_0
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor5
1while_lstm_cell_71_matmul_readvariableop_resource7
3while_lstm_cell_71_matmul_1_readvariableop_resource6
2while_lstm_cell_71_biasadd_readvariableop_resource¢)while/lstm_cell_71/BiasAdd/ReadVariableOp¢(while/lstm_cell_71/MatMul/ReadVariableOp¢*while/lstm_cell_71/MatMul_1/ReadVariableOpÃ
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   29
7while/TensorArrayV2Read/TensorListGetItem/element_shapeÓ
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItemÉ
(while/lstm_cell_71/MatMul/ReadVariableOpReadVariableOp3while_lstm_cell_71_matmul_readvariableop_resource_0*
_output_shapes
:	*
dtype02*
(while/lstm_cell_71/MatMul/ReadVariableOp×
while/lstm_cell_71/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:00while/lstm_cell_71/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/lstm_cell_71/MatMulÏ
*while/lstm_cell_71/MatMul_1/ReadVariableOpReadVariableOp5while_lstm_cell_71_matmul_1_readvariableop_resource_0*
_output_shapes
:	d*
dtype02,
*while/lstm_cell_71/MatMul_1/ReadVariableOpÀ
while/lstm_cell_71/MatMul_1MatMulwhile_placeholder_22while/lstm_cell_71/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/lstm_cell_71/MatMul_1¸
while/lstm_cell_71/addAddV2#while/lstm_cell_71/MatMul:product:0%while/lstm_cell_71/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/lstm_cell_71/addÈ
)while/lstm_cell_71/BiasAdd/ReadVariableOpReadVariableOp4while_lstm_cell_71_biasadd_readvariableop_resource_0*
_output_shapes	
:*
dtype02+
)while/lstm_cell_71/BiasAdd/ReadVariableOpÅ
while/lstm_cell_71/BiasAddBiasAddwhile/lstm_cell_71/add:z:01while/lstm_cell_71/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/lstm_cell_71/BiasAddv
while/lstm_cell_71/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
while/lstm_cell_71/Const
"while/lstm_cell_71/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2$
"while/lstm_cell_71/split/split_dim
while/lstm_cell_71/splitSplit+while/lstm_cell_71/split/split_dim:output:0#while/lstm_cell_71/BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd*
	num_split2
while/lstm_cell_71/split
while/lstm_cell_71/SigmoidSigmoid!while/lstm_cell_71/split:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd2
while/lstm_cell_71/Sigmoid
while/lstm_cell_71/Sigmoid_1Sigmoid!while/lstm_cell_71/split:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd2
while/lstm_cell_71/Sigmoid_1 
while/lstm_cell_71/mulMul while/lstm_cell_71/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd2
while/lstm_cell_71/mul
while/lstm_cell_71/ReluRelu!while/lstm_cell_71/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd2
while/lstm_cell_71/Relu´
while/lstm_cell_71/mul_1Mulwhile/lstm_cell_71/Sigmoid:y:0%while/lstm_cell_71/Relu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd2
while/lstm_cell_71/mul_1©
while/lstm_cell_71/add_1AddV2while/lstm_cell_71/mul:z:0while/lstm_cell_71/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd2
while/lstm_cell_71/add_1
while/lstm_cell_71/Sigmoid_2Sigmoid!while/lstm_cell_71/split:output:3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd2
while/lstm_cell_71/Sigmoid_2
while/lstm_cell_71/Relu_1Reluwhile/lstm_cell_71/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd2
while/lstm_cell_71/Relu_1¸
while/lstm_cell_71/mul_2Mul while/lstm_cell_71/Sigmoid_2:y:0'while/lstm_cell_71/Relu_1:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd2
while/lstm_cell_71/mul_2à
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_71/mul_2:z:0*
_output_shapes
: *
element_dtype02,
*while/TensorArrayV2Write/TensorListSetItem\
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add/yi
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: 2
	while/add`
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add_1/yv
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: 2
while/add_1â
while/IdentityIdentitywhile/add_1:z:0*^while/lstm_cell_71/BiasAdd/ReadVariableOp)^while/lstm_cell_71/MatMul/ReadVariableOp+^while/lstm_cell_71/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identityõ
while/Identity_1Identitywhile_while_maximum_iterations*^while/lstm_cell_71/BiasAdd/ReadVariableOp)^while/lstm_cell_71/MatMul/ReadVariableOp+^while/lstm_cell_71/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_1ä
while/Identity_2Identitywhile/add:z:0*^while/lstm_cell_71/BiasAdd/ReadVariableOp)^while/lstm_cell_71/MatMul/ReadVariableOp+^while/lstm_cell_71/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_2
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0*^while/lstm_cell_71/BiasAdd/ReadVariableOp)^while/lstm_cell_71/MatMul/ReadVariableOp+^while/lstm_cell_71/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_3
while/Identity_4Identitywhile/lstm_cell_71/mul_2:z:0*^while/lstm_cell_71/BiasAdd/ReadVariableOp)^while/lstm_cell_71/MatMul/ReadVariableOp+^while/lstm_cell_71/MatMul_1/ReadVariableOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd2
while/Identity_4
while/Identity_5Identitywhile/lstm_cell_71/add_1:z:0*^while/lstm_cell_71/BiasAdd/ReadVariableOp)^while/lstm_cell_71/MatMul/ReadVariableOp+^while/lstm_cell_71/MatMul_1/ReadVariableOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd2
while/Identity_5")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"j
2while_lstm_cell_71_biasadd_readvariableop_resource4while_lstm_cell_71_biasadd_readvariableop_resource_0"l
3while_lstm_cell_71_matmul_1_readvariableop_resource5while_lstm_cell_71_matmul_1_readvariableop_resource_0"h
1while_lstm_cell_71_matmul_readvariableop_resource3while_lstm_cell_71_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*Q
_input_shapes@
>: : : : :ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd: : :::2V
)while/lstm_cell_71/BiasAdd/ReadVariableOp)while/lstm_cell_71/BiasAdd/ReadVariableOp2T
(while/lstm_cell_71/MatMul/ReadVariableOp(while/lstm_cell_71/MatMul/ReadVariableOp2X
*while/lstm_cell_71/MatMul_1/ReadVariableOp*while/lstm_cell_71/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd:-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd:

_output_shapes
: :

_output_shapes
: 
«
Ã
while_cond_623975
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_14
0while_while_cond_623975___redundant_placeholder04
0while_while_cond_623975___redundant_placeholder14
0while_while_cond_623975___redundant_placeholder24
0while_while_cond_623975___redundant_placeholder3
while_identity
p

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: 2

while/Less]
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: 2
while/Identity")
while_identitywhile/Identity:output:0*S
_input_shapesB
@: : : : :ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd:-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd:

_output_shapes
: :

_output_shapes
:
Ï
°
.__inference_sequential_71_layer_call_fn_624708

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
identity¢StatefulPartitionedCall 
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3*
Tin

2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*'
_read_only_resource_inputs	
*-
config_proto

CPU

GPU 2J 8 *R
fMRK
I__inference_sequential_71_layer_call_and_return_conditional_losses_6243062
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:ÿÿÿÿÿÿÿÿÿ:::::22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
»
Í
-__inference_lstm_cell_71_layer_call_fn_625481

inputs
states_0
states_1
unknown
	unknown_0
	unknown_1
identity

identity_1

identity_2¢StatefulPartitionedCallÃ
StatefulPartitionedCallStatefulPartitionedCallinputsstates_0states_1unknown	unknown_0	unknown_1*
Tin

2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *Q
fLRJ
H__inference_lstm_cell_71_layer_call_and_return_conditional_losses_6233682
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd2

Identity

Identity_1Identity StatefulPartitionedCall:output:1^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd2

Identity_1

Identity_2Identity StatefulPartitionedCall:output:2^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd2

Identity_2"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*X
_input_shapesG
E:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd:::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs:QM
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
"
_user_specified_name
states/0:QM
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
"
_user_specified_name
states/1
«
Ã
while_cond_624128
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_14
0while_while_cond_624128___redundant_placeholder04
0while_while_cond_624128___redundant_placeholder14
0while_while_cond_624128___redundant_placeholder24
0while_while_cond_624128___redundant_placeholder3
while_identity
p

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: 2

while/Less]
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: 2
while/Identity")
while_identitywhile/Identity:output:0*S
_input_shapesB
@: : : : :ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd:-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd:

_output_shapes
: :

_output_shapes
:
»
Í
-__inference_lstm_cell_71_layer_call_fn_625498

inputs
states_0
states_1
unknown
	unknown_0
	unknown_1
identity

identity_1

identity_2¢StatefulPartitionedCallÃ
StatefulPartitionedCallStatefulPartitionedCallinputsstates_0states_1unknown	unknown_0	unknown_1*
Tin

2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *Q
fLRJ
H__inference_lstm_cell_71_layer_call_and_return_conditional_losses_6234012
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd2

Identity

Identity_1Identity StatefulPartitionedCall:output:1^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd2

Identity_1

Identity_2Identity StatefulPartitionedCall:output:2^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd2

Identity_2"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*X
_input_shapesG
E:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd:::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs:QM
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
"
_user_specified_name
states/0:QM
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
"
_user_specified_name
states/1
ÛD
Ü
C__inference_lstm_71_layer_call_and_return_conditional_losses_623896

inputs
lstm_cell_71_623814
lstm_cell_71_623816
lstm_cell_71_623818
identity¢$lstm_cell_71/StatefulPartitionedCall¢whileD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2â
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice\
zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :d2
zeros/mul/yl
	zeros/mulMulstrided_slice:output:0zeros/mul/y:output:0*
T0*
_output_shapes
: 2
	zeros/mul_
zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :è2
zeros/Less/yg

zeros/LessLesszeros/mul:z:0zeros/Less/y:output:0*
T0*
_output_shapes
: 2

zeros/Lessb
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :d2
zeros/packed/1
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros/packed_
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros/Constu
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd2
zeros`
zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B :d2
zeros_1/mul/yr
zeros_1/mulMulstrided_slice:output:0zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
zeros_1/mulc
zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :è2
zeros_1/Less/yo
zeros_1/LessLesszeros_1/mul:z:0zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
zeros_1/Lessf
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :d2
zeros_1/packed/1
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros_1/packedc
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros_1/Const}
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd2	
zeros_1u
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/perm
	transpose	Transposeinputstranspose/perm:output:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
	transposeO
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:2	
Shape_1x
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2î
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2
TensorArrayV2/element_shape²
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2¿
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   27
5TensorArrayUnstack/TensorListFromTensor/element_shapeø
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02)
'TensorArrayUnstack/TensorListFromTensorx
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2ü
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
shrink_axis_mask2
strided_slice_2
$lstm_cell_71/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_2:output:0zeros:output:0zeros_1:output:0lstm_cell_71_623814lstm_cell_71_623816lstm_cell_71_623818*
Tin

2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *Q
fLRJ
H__inference_lstm_cell_71_layer_call_and_return_conditional_losses_6234012&
$lstm_cell_71/StatefulPartitionedCall
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿd   2
TensorArrayV2_1/element_shape¸
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2_1N
timeConst*
_output_shapes
: *
dtype0*
value	B : 2
time
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counter£
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0lstm_cell_71_623814lstm_cell_71_623816lstm_cell_71_623818*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd: : : : : *%
_read_only_resource_inputs
	
*
bodyR
while_body_623827*
condR
while_cond_623826*K
output_shapes:
8: : : : :ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd: : : : : *
parallel_iterations 2
whileµ
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿd   22
0TensorArrayV2Stack/TensorListStack/element_shapeñ
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿd*
element_dtype02$
"TensorArrayV2Stack/TensorListStack
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿ2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd*
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/perm®
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿd2
transpose_1f
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2	
runtime
IdentityIdentitystrided_slice_3:output:0%^lstm_cell_71/StatefulPartitionedCall^while*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:::2L
$lstm_cell_71/StatefulPartitionedCall$lstm_cell_71/StatefulPartitionedCall2
whilewhile:\ X
4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs


(__inference_lstm_71_layer_call_fn_625368

inputs
unknown
	unknown_0
	unknown_1
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_lstm_71_layer_call_and_return_conditional_losses_6240612
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:ÿÿÿÿÿÿÿÿÿ:::22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs


(__inference_lstm_71_layer_call_fn_625379

inputs
unknown
	unknown_0
	unknown_1
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_lstm_71_layer_call_and_return_conditional_losses_6242142
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:ÿÿÿÿÿÿÿÿÿ:::22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ñ
Ç
!__inference__wrapped_model_623295
lstm_71_inputE
Asequential_71_lstm_71_lstm_cell_71_matmul_readvariableop_resourceG
Csequential_71_lstm_71_lstm_cell_71_matmul_1_readvariableop_resourceF
Bsequential_71_lstm_71_lstm_cell_71_biasadd_readvariableop_resource9
5sequential_71_dense_71_matmul_readvariableop_resource:
6sequential_71_dense_71_biasadd_readvariableop_resource
identity¢-sequential_71/dense_71/BiasAdd/ReadVariableOp¢,sequential_71/dense_71/MatMul/ReadVariableOp¢9sequential_71/lstm_71/lstm_cell_71/BiasAdd/ReadVariableOp¢8sequential_71/lstm_71/lstm_cell_71/MatMul/ReadVariableOp¢:sequential_71/lstm_71/lstm_cell_71/MatMul_1/ReadVariableOp¢sequential_71/lstm_71/whilew
sequential_71/lstm_71/ShapeShapelstm_71_input*
T0*
_output_shapes
:2
sequential_71/lstm_71/Shape 
)sequential_71/lstm_71/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2+
)sequential_71/lstm_71/strided_slice/stack¤
+sequential_71/lstm_71/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2-
+sequential_71/lstm_71/strided_slice/stack_1¤
+sequential_71/lstm_71/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2-
+sequential_71/lstm_71/strided_slice/stack_2æ
#sequential_71/lstm_71/strided_sliceStridedSlice$sequential_71/lstm_71/Shape:output:02sequential_71/lstm_71/strided_slice/stack:output:04sequential_71/lstm_71/strided_slice/stack_1:output:04sequential_71/lstm_71/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2%
#sequential_71/lstm_71/strided_slice
!sequential_71/lstm_71/zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :d2#
!sequential_71/lstm_71/zeros/mul/yÄ
sequential_71/lstm_71/zeros/mulMul,sequential_71/lstm_71/strided_slice:output:0*sequential_71/lstm_71/zeros/mul/y:output:0*
T0*
_output_shapes
: 2!
sequential_71/lstm_71/zeros/mul
"sequential_71/lstm_71/zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :è2$
"sequential_71/lstm_71/zeros/Less/y¿
 sequential_71/lstm_71/zeros/LessLess#sequential_71/lstm_71/zeros/mul:z:0+sequential_71/lstm_71/zeros/Less/y:output:0*
T0*
_output_shapes
: 2"
 sequential_71/lstm_71/zeros/Less
$sequential_71/lstm_71/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :d2&
$sequential_71/lstm_71/zeros/packed/1Û
"sequential_71/lstm_71/zeros/packedPack,sequential_71/lstm_71/strided_slice:output:0-sequential_71/lstm_71/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2$
"sequential_71/lstm_71/zeros/packed
!sequential_71/lstm_71/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2#
!sequential_71/lstm_71/zeros/ConstÍ
sequential_71/lstm_71/zerosFill+sequential_71/lstm_71/zeros/packed:output:0*sequential_71/lstm_71/zeros/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd2
sequential_71/lstm_71/zeros
#sequential_71/lstm_71/zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B :d2%
#sequential_71/lstm_71/zeros_1/mul/yÊ
!sequential_71/lstm_71/zeros_1/mulMul,sequential_71/lstm_71/strided_slice:output:0,sequential_71/lstm_71/zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2#
!sequential_71/lstm_71/zeros_1/mul
$sequential_71/lstm_71/zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :è2&
$sequential_71/lstm_71/zeros_1/Less/yÇ
"sequential_71/lstm_71/zeros_1/LessLess%sequential_71/lstm_71/zeros_1/mul:z:0-sequential_71/lstm_71/zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2$
"sequential_71/lstm_71/zeros_1/Less
&sequential_71/lstm_71/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :d2(
&sequential_71/lstm_71/zeros_1/packed/1á
$sequential_71/lstm_71/zeros_1/packedPack,sequential_71/lstm_71/strided_slice:output:0/sequential_71/lstm_71/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2&
$sequential_71/lstm_71/zeros_1/packed
#sequential_71/lstm_71/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2%
#sequential_71/lstm_71/zeros_1/ConstÕ
sequential_71/lstm_71/zeros_1Fill-sequential_71/lstm_71/zeros_1/packed:output:0,sequential_71/lstm_71/zeros_1/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd2
sequential_71/lstm_71/zeros_1¡
$sequential_71/lstm_71/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2&
$sequential_71/lstm_71/transpose/permÃ
sequential_71/lstm_71/transpose	Transposelstm_71_input-sequential_71/lstm_71/transpose/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2!
sequential_71/lstm_71/transpose
sequential_71/lstm_71/Shape_1Shape#sequential_71/lstm_71/transpose:y:0*
T0*
_output_shapes
:2
sequential_71/lstm_71/Shape_1¤
+sequential_71/lstm_71/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2-
+sequential_71/lstm_71/strided_slice_1/stack¨
-sequential_71/lstm_71/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2/
-sequential_71/lstm_71/strided_slice_1/stack_1¨
-sequential_71/lstm_71/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2/
-sequential_71/lstm_71/strided_slice_1/stack_2ò
%sequential_71/lstm_71/strided_slice_1StridedSlice&sequential_71/lstm_71/Shape_1:output:04sequential_71/lstm_71/strided_slice_1/stack:output:06sequential_71/lstm_71/strided_slice_1/stack_1:output:06sequential_71/lstm_71/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2'
%sequential_71/lstm_71/strided_slice_1±
1sequential_71/lstm_71/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ23
1sequential_71/lstm_71/TensorArrayV2/element_shape
#sequential_71/lstm_71/TensorArrayV2TensorListReserve:sequential_71/lstm_71/TensorArrayV2/element_shape:output:0.sequential_71/lstm_71/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02%
#sequential_71/lstm_71/TensorArrayV2ë
Ksequential_71/lstm_71/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   2M
Ksequential_71/lstm_71/TensorArrayUnstack/TensorListFromTensor/element_shapeÐ
=sequential_71/lstm_71/TensorArrayUnstack/TensorListFromTensorTensorListFromTensor#sequential_71/lstm_71/transpose:y:0Tsequential_71/lstm_71/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02?
=sequential_71/lstm_71/TensorArrayUnstack/TensorListFromTensor¤
+sequential_71/lstm_71/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2-
+sequential_71/lstm_71/strided_slice_2/stack¨
-sequential_71/lstm_71/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2/
-sequential_71/lstm_71/strided_slice_2/stack_1¨
-sequential_71/lstm_71/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2/
-sequential_71/lstm_71/strided_slice_2/stack_2
%sequential_71/lstm_71/strided_slice_2StridedSlice#sequential_71/lstm_71/transpose:y:04sequential_71/lstm_71/strided_slice_2/stack:output:06sequential_71/lstm_71/strided_slice_2/stack_1:output:06sequential_71/lstm_71/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
shrink_axis_mask2'
%sequential_71/lstm_71/strided_slice_2÷
8sequential_71/lstm_71/lstm_cell_71/MatMul/ReadVariableOpReadVariableOpAsequential_71_lstm_71_lstm_cell_71_matmul_readvariableop_resource*
_output_shapes
:	*
dtype02:
8sequential_71/lstm_71/lstm_cell_71/MatMul/ReadVariableOp
)sequential_71/lstm_71/lstm_cell_71/MatMulMatMul.sequential_71/lstm_71/strided_slice_2:output:0@sequential_71/lstm_71/lstm_cell_71/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2+
)sequential_71/lstm_71/lstm_cell_71/MatMulý
:sequential_71/lstm_71/lstm_cell_71/MatMul_1/ReadVariableOpReadVariableOpCsequential_71_lstm_71_lstm_cell_71_matmul_1_readvariableop_resource*
_output_shapes
:	d*
dtype02<
:sequential_71/lstm_71/lstm_cell_71/MatMul_1/ReadVariableOp
+sequential_71/lstm_71/lstm_cell_71/MatMul_1MatMul$sequential_71/lstm_71/zeros:output:0Bsequential_71/lstm_71/lstm_cell_71/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2-
+sequential_71/lstm_71/lstm_cell_71/MatMul_1ø
&sequential_71/lstm_71/lstm_cell_71/addAddV23sequential_71/lstm_71/lstm_cell_71/MatMul:product:05sequential_71/lstm_71/lstm_cell_71/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2(
&sequential_71/lstm_71/lstm_cell_71/addö
9sequential_71/lstm_71/lstm_cell_71/BiasAdd/ReadVariableOpReadVariableOpBsequential_71_lstm_71_lstm_cell_71_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02;
9sequential_71/lstm_71/lstm_cell_71/BiasAdd/ReadVariableOp
*sequential_71/lstm_71/lstm_cell_71/BiasAddBiasAdd*sequential_71/lstm_71/lstm_cell_71/add:z:0Asequential_71/lstm_71/lstm_cell_71/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2,
*sequential_71/lstm_71/lstm_cell_71/BiasAdd
(sequential_71/lstm_71/lstm_cell_71/ConstConst*
_output_shapes
: *
dtype0*
value	B :2*
(sequential_71/lstm_71/lstm_cell_71/Constª
2sequential_71/lstm_71/lstm_cell_71/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :24
2sequential_71/lstm_71/lstm_cell_71/split/split_dimË
(sequential_71/lstm_71/lstm_cell_71/splitSplit;sequential_71/lstm_71/lstm_cell_71/split/split_dim:output:03sequential_71/lstm_71/lstm_cell_71/BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd*
	num_split2*
(sequential_71/lstm_71/lstm_cell_71/splitÈ
*sequential_71/lstm_71/lstm_cell_71/SigmoidSigmoid1sequential_71/lstm_71/lstm_cell_71/split:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd2,
*sequential_71/lstm_71/lstm_cell_71/SigmoidÌ
,sequential_71/lstm_71/lstm_cell_71/Sigmoid_1Sigmoid1sequential_71/lstm_71/lstm_cell_71/split:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd2.
,sequential_71/lstm_71/lstm_cell_71/Sigmoid_1ã
&sequential_71/lstm_71/lstm_cell_71/mulMul0sequential_71/lstm_71/lstm_cell_71/Sigmoid_1:y:0&sequential_71/lstm_71/zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd2(
&sequential_71/lstm_71/lstm_cell_71/mul¿
'sequential_71/lstm_71/lstm_cell_71/ReluRelu1sequential_71/lstm_71/lstm_cell_71/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd2)
'sequential_71/lstm_71/lstm_cell_71/Reluô
(sequential_71/lstm_71/lstm_cell_71/mul_1Mul.sequential_71/lstm_71/lstm_cell_71/Sigmoid:y:05sequential_71/lstm_71/lstm_cell_71/Relu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd2*
(sequential_71/lstm_71/lstm_cell_71/mul_1é
(sequential_71/lstm_71/lstm_cell_71/add_1AddV2*sequential_71/lstm_71/lstm_cell_71/mul:z:0,sequential_71/lstm_71/lstm_cell_71/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd2*
(sequential_71/lstm_71/lstm_cell_71/add_1Ì
,sequential_71/lstm_71/lstm_cell_71/Sigmoid_2Sigmoid1sequential_71/lstm_71/lstm_cell_71/split:output:3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd2.
,sequential_71/lstm_71/lstm_cell_71/Sigmoid_2¾
)sequential_71/lstm_71/lstm_cell_71/Relu_1Relu,sequential_71/lstm_71/lstm_cell_71/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd2+
)sequential_71/lstm_71/lstm_cell_71/Relu_1ø
(sequential_71/lstm_71/lstm_cell_71/mul_2Mul0sequential_71/lstm_71/lstm_cell_71/Sigmoid_2:y:07sequential_71/lstm_71/lstm_cell_71/Relu_1:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd2*
(sequential_71/lstm_71/lstm_cell_71/mul_2»
3sequential_71/lstm_71/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿd   25
3sequential_71/lstm_71/TensorArrayV2_1/element_shape
%sequential_71/lstm_71/TensorArrayV2_1TensorListReserve<sequential_71/lstm_71/TensorArrayV2_1/element_shape:output:0.sequential_71/lstm_71/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02'
%sequential_71/lstm_71/TensorArrayV2_1z
sequential_71/lstm_71/timeConst*
_output_shapes
: *
dtype0*
value	B : 2
sequential_71/lstm_71/time«
.sequential_71/lstm_71/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ20
.sequential_71/lstm_71/while/maximum_iterations
(sequential_71/lstm_71/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2*
(sequential_71/lstm_71/while/loop_counter¸
sequential_71/lstm_71/whileWhile1sequential_71/lstm_71/while/loop_counter:output:07sequential_71/lstm_71/while/maximum_iterations:output:0#sequential_71/lstm_71/time:output:0.sequential_71/lstm_71/TensorArrayV2_1:handle:0$sequential_71/lstm_71/zeros:output:0&sequential_71/lstm_71/zeros_1:output:0.sequential_71/lstm_71/strided_slice_1:output:0Msequential_71/lstm_71/TensorArrayUnstack/TensorListFromTensor:output_handle:0Asequential_71_lstm_71_lstm_cell_71_matmul_readvariableop_resourceCsequential_71_lstm_71_lstm_cell_71_matmul_1_readvariableop_resourceBsequential_71_lstm_71_lstm_cell_71_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd: : : : : *%
_read_only_resource_inputs
	
*3
body+R)
'sequential_71_lstm_71_while_body_623204*3
cond+R)
'sequential_71_lstm_71_while_cond_623203*K
output_shapes:
8: : : : :ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd: : : : : *
parallel_iterations 2
sequential_71/lstm_71/whileá
Fsequential_71/lstm_71/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿd   2H
Fsequential_71/lstm_71/TensorArrayV2Stack/TensorListStack/element_shapeÀ
8sequential_71/lstm_71/TensorArrayV2Stack/TensorListStackTensorListStack$sequential_71/lstm_71/while:output:3Osequential_71/lstm_71/TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿd*
element_dtype02:
8sequential_71/lstm_71/TensorArrayV2Stack/TensorListStack­
+sequential_71/lstm_71/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿ2-
+sequential_71/lstm_71/strided_slice_3/stack¨
-sequential_71/lstm_71/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2/
-sequential_71/lstm_71/strided_slice_3/stack_1¨
-sequential_71/lstm_71/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2/
-sequential_71/lstm_71/strided_slice_3/stack_2
%sequential_71/lstm_71/strided_slice_3StridedSliceAsequential_71/lstm_71/TensorArrayV2Stack/TensorListStack:tensor:04sequential_71/lstm_71/strided_slice_3/stack:output:06sequential_71/lstm_71/strided_slice_3/stack_1:output:06sequential_71/lstm_71/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd*
shrink_axis_mask2'
%sequential_71/lstm_71/strided_slice_3¥
&sequential_71/lstm_71/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2(
&sequential_71/lstm_71/transpose_1/permý
!sequential_71/lstm_71/transpose_1	TransposeAsequential_71/lstm_71/TensorArrayV2Stack/TensorListStack:tensor:0/sequential_71/lstm_71/transpose_1/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿd2#
!sequential_71/lstm_71/transpose_1
sequential_71/lstm_71/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2
sequential_71/lstm_71/runtimeÒ
,sequential_71/dense_71/MatMul/ReadVariableOpReadVariableOp5sequential_71_dense_71_matmul_readvariableop_resource*
_output_shapes

:d*
dtype02.
,sequential_71/dense_71/MatMul/ReadVariableOpà
sequential_71/dense_71/MatMulMatMul.sequential_71/lstm_71/strided_slice_3:output:04sequential_71/dense_71/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
sequential_71/dense_71/MatMulÑ
-sequential_71/dense_71/BiasAdd/ReadVariableOpReadVariableOp6sequential_71_dense_71_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02/
-sequential_71/dense_71/BiasAdd/ReadVariableOpÝ
sequential_71/dense_71/BiasAddBiasAdd'sequential_71/dense_71/MatMul:product:05sequential_71/dense_71/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2 
sequential_71/dense_71/BiasAdd¬
IdentityIdentity'sequential_71/dense_71/BiasAdd:output:0.^sequential_71/dense_71/BiasAdd/ReadVariableOp-^sequential_71/dense_71/MatMul/ReadVariableOp:^sequential_71/lstm_71/lstm_cell_71/BiasAdd/ReadVariableOp9^sequential_71/lstm_71/lstm_cell_71/MatMul/ReadVariableOp;^sequential_71/lstm_71/lstm_cell_71/MatMul_1/ReadVariableOp^sequential_71/lstm_71/while*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:ÿÿÿÿÿÿÿÿÿ:::::2^
-sequential_71/dense_71/BiasAdd/ReadVariableOp-sequential_71/dense_71/BiasAdd/ReadVariableOp2\
,sequential_71/dense_71/MatMul/ReadVariableOp,sequential_71/dense_71/MatMul/ReadVariableOp2v
9sequential_71/lstm_71/lstm_cell_71/BiasAdd/ReadVariableOp9sequential_71/lstm_71/lstm_cell_71/BiasAdd/ReadVariableOp2t
8sequential_71/lstm_71/lstm_cell_71/MatMul/ReadVariableOp8sequential_71/lstm_71/lstm_cell_71/MatMul/ReadVariableOp2x
:sequential_71/lstm_71/lstm_cell_71/MatMul_1/ReadVariableOp:sequential_71/lstm_71/lstm_cell_71/MatMul_1/ReadVariableOp2:
sequential_71/lstm_71/whilesequential_71/lstm_71/while:Z V
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
'
_user_specified_namelstm_71_input
×r
Î
I__inference_sequential_71_layer_call_and_return_conditional_losses_624693

inputs7
3lstm_71_lstm_cell_71_matmul_readvariableop_resource9
5lstm_71_lstm_cell_71_matmul_1_readvariableop_resource8
4lstm_71_lstm_cell_71_biasadd_readvariableop_resource+
'dense_71_matmul_readvariableop_resource,
(dense_71_biasadd_readvariableop_resource
identity¢dense_71/BiasAdd/ReadVariableOp¢dense_71/MatMul/ReadVariableOp¢+lstm_71/lstm_cell_71/BiasAdd/ReadVariableOp¢*lstm_71/lstm_cell_71/MatMul/ReadVariableOp¢,lstm_71/lstm_cell_71/MatMul_1/ReadVariableOp¢lstm_71/whileT
lstm_71/ShapeShapeinputs*
T0*
_output_shapes
:2
lstm_71/Shape
lstm_71/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
lstm_71/strided_slice/stack
lstm_71/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
lstm_71/strided_slice/stack_1
lstm_71/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
lstm_71/strided_slice/stack_2
lstm_71/strided_sliceStridedSlicelstm_71/Shape:output:0$lstm_71/strided_slice/stack:output:0&lstm_71/strided_slice/stack_1:output:0&lstm_71/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
lstm_71/strided_slicel
lstm_71/zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :d2
lstm_71/zeros/mul/y
lstm_71/zeros/mulMullstm_71/strided_slice:output:0lstm_71/zeros/mul/y:output:0*
T0*
_output_shapes
: 2
lstm_71/zeros/mulo
lstm_71/zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :è2
lstm_71/zeros/Less/y
lstm_71/zeros/LessLesslstm_71/zeros/mul:z:0lstm_71/zeros/Less/y:output:0*
T0*
_output_shapes
: 2
lstm_71/zeros/Lessr
lstm_71/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :d2
lstm_71/zeros/packed/1£
lstm_71/zeros/packedPacklstm_71/strided_slice:output:0lstm_71/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
lstm_71/zeros/packedo
lstm_71/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
lstm_71/zeros/Const
lstm_71/zerosFilllstm_71/zeros/packed:output:0lstm_71/zeros/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd2
lstm_71/zerosp
lstm_71/zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B :d2
lstm_71/zeros_1/mul/y
lstm_71/zeros_1/mulMullstm_71/strided_slice:output:0lstm_71/zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
lstm_71/zeros_1/muls
lstm_71/zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :è2
lstm_71/zeros_1/Less/y
lstm_71/zeros_1/LessLesslstm_71/zeros_1/mul:z:0lstm_71/zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
lstm_71/zeros_1/Lessv
lstm_71/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :d2
lstm_71/zeros_1/packed/1©
lstm_71/zeros_1/packedPacklstm_71/strided_slice:output:0!lstm_71/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
lstm_71/zeros_1/packeds
lstm_71/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
lstm_71/zeros_1/Const
lstm_71/zeros_1Filllstm_71/zeros_1/packed:output:0lstm_71/zeros_1/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd2
lstm_71/zeros_1
lstm_71/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
lstm_71/transpose/perm
lstm_71/transpose	Transposeinputslstm_71/transpose/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_71/transposeg
lstm_71/Shape_1Shapelstm_71/transpose:y:0*
T0*
_output_shapes
:2
lstm_71/Shape_1
lstm_71/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
lstm_71/strided_slice_1/stack
lstm_71/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2!
lstm_71/strided_slice_1/stack_1
lstm_71/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2!
lstm_71/strided_slice_1/stack_2
lstm_71/strided_slice_1StridedSlicelstm_71/Shape_1:output:0&lstm_71/strided_slice_1/stack:output:0(lstm_71/strided_slice_1/stack_1:output:0(lstm_71/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
lstm_71/strided_slice_1
#lstm_71/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2%
#lstm_71/TensorArrayV2/element_shapeÒ
lstm_71/TensorArrayV2TensorListReserve,lstm_71/TensorArrayV2/element_shape:output:0 lstm_71/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
lstm_71/TensorArrayV2Ï
=lstm_71/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   2?
=lstm_71/TensorArrayUnstack/TensorListFromTensor/element_shape
/lstm_71/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorlstm_71/transpose:y:0Flstm_71/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type021
/lstm_71/TensorArrayUnstack/TensorListFromTensor
lstm_71/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
lstm_71/strided_slice_2/stack
lstm_71/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2!
lstm_71/strided_slice_2/stack_1
lstm_71/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2!
lstm_71/strided_slice_2/stack_2¬
lstm_71/strided_slice_2StridedSlicelstm_71/transpose:y:0&lstm_71/strided_slice_2/stack:output:0(lstm_71/strided_slice_2/stack_1:output:0(lstm_71/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
shrink_axis_mask2
lstm_71/strided_slice_2Í
*lstm_71/lstm_cell_71/MatMul/ReadVariableOpReadVariableOp3lstm_71_lstm_cell_71_matmul_readvariableop_resource*
_output_shapes
:	*
dtype02,
*lstm_71/lstm_cell_71/MatMul/ReadVariableOpÍ
lstm_71/lstm_cell_71/MatMulMatMul lstm_71/strided_slice_2:output:02lstm_71/lstm_cell_71/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_71/lstm_cell_71/MatMulÓ
,lstm_71/lstm_cell_71/MatMul_1/ReadVariableOpReadVariableOp5lstm_71_lstm_cell_71_matmul_1_readvariableop_resource*
_output_shapes
:	d*
dtype02.
,lstm_71/lstm_cell_71/MatMul_1/ReadVariableOpÉ
lstm_71/lstm_cell_71/MatMul_1MatMullstm_71/zeros:output:04lstm_71/lstm_cell_71/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_71/lstm_cell_71/MatMul_1À
lstm_71/lstm_cell_71/addAddV2%lstm_71/lstm_cell_71/MatMul:product:0'lstm_71/lstm_cell_71/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_71/lstm_cell_71/addÌ
+lstm_71/lstm_cell_71/BiasAdd/ReadVariableOpReadVariableOp4lstm_71_lstm_cell_71_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02-
+lstm_71/lstm_cell_71/BiasAdd/ReadVariableOpÍ
lstm_71/lstm_cell_71/BiasAddBiasAddlstm_71/lstm_cell_71/add:z:03lstm_71/lstm_cell_71/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_71/lstm_cell_71/BiasAddz
lstm_71/lstm_cell_71/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_71/lstm_cell_71/Const
$lstm_71/lstm_cell_71/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2&
$lstm_71/lstm_cell_71/split/split_dim
lstm_71/lstm_cell_71/splitSplit-lstm_71/lstm_cell_71/split/split_dim:output:0%lstm_71/lstm_cell_71/BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd*
	num_split2
lstm_71/lstm_cell_71/split
lstm_71/lstm_cell_71/SigmoidSigmoid#lstm_71/lstm_cell_71/split:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd2
lstm_71/lstm_cell_71/Sigmoid¢
lstm_71/lstm_cell_71/Sigmoid_1Sigmoid#lstm_71/lstm_cell_71/split:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd2 
lstm_71/lstm_cell_71/Sigmoid_1«
lstm_71/lstm_cell_71/mulMul"lstm_71/lstm_cell_71/Sigmoid_1:y:0lstm_71/zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd2
lstm_71/lstm_cell_71/mul
lstm_71/lstm_cell_71/ReluRelu#lstm_71/lstm_cell_71/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd2
lstm_71/lstm_cell_71/Relu¼
lstm_71/lstm_cell_71/mul_1Mul lstm_71/lstm_cell_71/Sigmoid:y:0'lstm_71/lstm_cell_71/Relu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd2
lstm_71/lstm_cell_71/mul_1±
lstm_71/lstm_cell_71/add_1AddV2lstm_71/lstm_cell_71/mul:z:0lstm_71/lstm_cell_71/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd2
lstm_71/lstm_cell_71/add_1¢
lstm_71/lstm_cell_71/Sigmoid_2Sigmoid#lstm_71/lstm_cell_71/split:output:3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd2 
lstm_71/lstm_cell_71/Sigmoid_2
lstm_71/lstm_cell_71/Relu_1Relulstm_71/lstm_cell_71/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd2
lstm_71/lstm_cell_71/Relu_1À
lstm_71/lstm_cell_71/mul_2Mul"lstm_71/lstm_cell_71/Sigmoid_2:y:0)lstm_71/lstm_cell_71/Relu_1:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd2
lstm_71/lstm_cell_71/mul_2
%lstm_71/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿd   2'
%lstm_71/TensorArrayV2_1/element_shapeØ
lstm_71/TensorArrayV2_1TensorListReserve.lstm_71/TensorArrayV2_1/element_shape:output:0 lstm_71/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
lstm_71/TensorArrayV2_1^
lstm_71/timeConst*
_output_shapes
: *
dtype0*
value	B : 2
lstm_71/time
 lstm_71/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2"
 lstm_71/while/maximum_iterationsz
lstm_71/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
lstm_71/while/loop_counteræ
lstm_71/whileWhile#lstm_71/while/loop_counter:output:0)lstm_71/while/maximum_iterations:output:0lstm_71/time:output:0 lstm_71/TensorArrayV2_1:handle:0lstm_71/zeros:output:0lstm_71/zeros_1:output:0 lstm_71/strided_slice_1:output:0?lstm_71/TensorArrayUnstack/TensorListFromTensor:output_handle:03lstm_71_lstm_cell_71_matmul_readvariableop_resource5lstm_71_lstm_cell_71_matmul_1_readvariableop_resource4lstm_71_lstm_cell_71_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd: : : : : *%
_read_only_resource_inputs
	
*%
bodyR
lstm_71_while_body_624602*%
condR
lstm_71_while_cond_624601*K
output_shapes:
8: : : : :ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd: : : : : *
parallel_iterations 2
lstm_71/whileÅ
8lstm_71/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿd   2:
8lstm_71/TensorArrayV2Stack/TensorListStack/element_shape
*lstm_71/TensorArrayV2Stack/TensorListStackTensorListStacklstm_71/while:output:3Alstm_71/TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿd*
element_dtype02,
*lstm_71/TensorArrayV2Stack/TensorListStack
lstm_71/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿ2
lstm_71/strided_slice_3/stack
lstm_71/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2!
lstm_71/strided_slice_3/stack_1
lstm_71/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2!
lstm_71/strided_slice_3/stack_2Ê
lstm_71/strided_slice_3StridedSlice3lstm_71/TensorArrayV2Stack/TensorListStack:tensor:0&lstm_71/strided_slice_3/stack:output:0(lstm_71/strided_slice_3/stack_1:output:0(lstm_71/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd*
shrink_axis_mask2
lstm_71/strided_slice_3
lstm_71/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
lstm_71/transpose_1/permÅ
lstm_71/transpose_1	Transpose3lstm_71/TensorArrayV2Stack/TensorListStack:tensor:0!lstm_71/transpose_1/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿd2
lstm_71/transpose_1v
lstm_71/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2
lstm_71/runtime¨
dense_71/MatMul/ReadVariableOpReadVariableOp'dense_71_matmul_readvariableop_resource*
_output_shapes

:d*
dtype02 
dense_71/MatMul/ReadVariableOp¨
dense_71/MatMulMatMul lstm_71/strided_slice_3:output:0&dense_71/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_71/MatMul§
dense_71/BiasAdd/ReadVariableOpReadVariableOp(dense_71_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
dense_71/BiasAdd/ReadVariableOp¥
dense_71/BiasAddBiasAdddense_71/MatMul:product:0'dense_71/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_71/BiasAddÊ
IdentityIdentitydense_71/BiasAdd:output:0 ^dense_71/BiasAdd/ReadVariableOp^dense_71/MatMul/ReadVariableOp,^lstm_71/lstm_cell_71/BiasAdd/ReadVariableOp+^lstm_71/lstm_cell_71/MatMul/ReadVariableOp-^lstm_71/lstm_cell_71/MatMul_1/ReadVariableOp^lstm_71/while*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:ÿÿÿÿÿÿÿÿÿ:::::2B
dense_71/BiasAdd/ReadVariableOpdense_71/BiasAdd/ReadVariableOp2@
dense_71/MatMul/ReadVariableOpdense_71/MatMul/ReadVariableOp2Z
+lstm_71/lstm_cell_71/BiasAdd/ReadVariableOp+lstm_71/lstm_cell_71/BiasAdd/ReadVariableOp2X
*lstm_71/lstm_cell_71/MatMul/ReadVariableOp*lstm_71/lstm_cell_71/MatMul/ReadVariableOp2\
,lstm_71/lstm_cell_71/MatMul_1/ReadVariableOp,lstm_71/lstm_cell_71/MatMul_1/ReadVariableOp2
lstm_71/whilelstm_71/while:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
³
Ý
H__inference_lstm_cell_71_layer_call_and_return_conditional_losses_625431

inputs
states_0
states_1"
matmul_readvariableop_resource$
 matmul_1_readvariableop_resource#
biasadd_readvariableop_resource
identity

identity_1

identity_2¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp¢MatMul_1/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
MatMul
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource*
_output_shapes
:	d*
dtype02
MatMul_1/ReadVariableOp|
MatMul_1MatMulstates_0MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

MatMul_1l
addAddV2MatMul:product:0MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
add
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype02
BiasAdd/ReadVariableOpy
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
BiasAddP
ConstConst*
_output_shapes
: *
dtype0*
value	B :2
Constd
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split/split_dim¿
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd*
	num_split2
split_
SigmoidSigmoidsplit:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd2	
Sigmoidc
	Sigmoid_1Sigmoidsplit:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd2
	Sigmoid_1\
mulMulSigmoid_1:y:0states_1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd2
mulV
ReluRelusplit:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd2
Reluh
mul_1MulSigmoid:y:0Relu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd2
mul_1]
add_1AddV2mul:z:0	mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd2
add_1c
	Sigmoid_2Sigmoidsplit:output:3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd2
	Sigmoid_2U
Relu_1Relu	add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd2
Relu_1l
mul_2MulSigmoid_2:y:0Relu_1:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd2
mul_2¨
IdentityIdentity	mul_2:z:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd2

Identity¬

Identity_1Identity	mul_2:z:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd2

Identity_1¬

Identity_2Identity	add_1:z:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd2

Identity_2"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*X
_input_shapesG
E:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd:::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs:QM
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
"
_user_specified_name
states/0:QM
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
"
_user_specified_name
states/1
[
ò
C__inference_lstm_71_layer_call_and_return_conditional_losses_625357

inputs/
+lstm_cell_71_matmul_readvariableop_resource1
-lstm_cell_71_matmul_1_readvariableop_resource0
,lstm_cell_71_biasadd_readvariableop_resource
identity¢#lstm_cell_71/BiasAdd/ReadVariableOp¢"lstm_cell_71/MatMul/ReadVariableOp¢$lstm_cell_71/MatMul_1/ReadVariableOp¢whileD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2â
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice\
zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :d2
zeros/mul/yl
	zeros/mulMulstrided_slice:output:0zeros/mul/y:output:0*
T0*
_output_shapes
: 2
	zeros/mul_
zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :è2
zeros/Less/yg

zeros/LessLesszeros/mul:z:0zeros/Less/y:output:0*
T0*
_output_shapes
: 2

zeros/Lessb
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :d2
zeros/packed/1
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros/packed_
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros/Constu
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd2
zeros`
zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B :d2
zeros_1/mul/yr
zeros_1/mulMulstrided_slice:output:0zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
zeros_1/mulc
zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :è2
zeros_1/Less/yo
zeros_1/LessLesszeros_1/mul:z:0zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
zeros_1/Lessf
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :d2
zeros_1/packed/1
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros_1/packedc
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros_1/Const}
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd2	
zeros_1u
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/permz
	transpose	Transposeinputstranspose/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
	transposeO
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:2	
Shape_1x
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2î
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2
TensorArrayV2/element_shape²
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2¿
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   27
5TensorArrayUnstack/TensorListFromTensor/element_shapeø
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02)
'TensorArrayUnstack/TensorListFromTensorx
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2ü
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
shrink_axis_mask2
strided_slice_2µ
"lstm_cell_71/MatMul/ReadVariableOpReadVariableOp+lstm_cell_71_matmul_readvariableop_resource*
_output_shapes
:	*
dtype02$
"lstm_cell_71/MatMul/ReadVariableOp­
lstm_cell_71/MatMulMatMulstrided_slice_2:output:0*lstm_cell_71/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell_71/MatMul»
$lstm_cell_71/MatMul_1/ReadVariableOpReadVariableOp-lstm_cell_71_matmul_1_readvariableop_resource*
_output_shapes
:	d*
dtype02&
$lstm_cell_71/MatMul_1/ReadVariableOp©
lstm_cell_71/MatMul_1MatMulzeros:output:0,lstm_cell_71/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell_71/MatMul_1 
lstm_cell_71/addAddV2lstm_cell_71/MatMul:product:0lstm_cell_71/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell_71/add´
#lstm_cell_71/BiasAdd/ReadVariableOpReadVariableOp,lstm_cell_71_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02%
#lstm_cell_71/BiasAdd/ReadVariableOp­
lstm_cell_71/BiasAddBiasAddlstm_cell_71/add:z:0+lstm_cell_71/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell_71/BiasAddj
lstm_cell_71/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_71/Const~
lstm_cell_71/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_71/split/split_dimó
lstm_cell_71/splitSplit%lstm_cell_71/split/split_dim:output:0lstm_cell_71/BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd*
	num_split2
lstm_cell_71/split
lstm_cell_71/SigmoidSigmoidlstm_cell_71/split:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd2
lstm_cell_71/Sigmoid
lstm_cell_71/Sigmoid_1Sigmoidlstm_cell_71/split:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd2
lstm_cell_71/Sigmoid_1
lstm_cell_71/mulMullstm_cell_71/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd2
lstm_cell_71/mul}
lstm_cell_71/ReluRelulstm_cell_71/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd2
lstm_cell_71/Relu
lstm_cell_71/mul_1Mullstm_cell_71/Sigmoid:y:0lstm_cell_71/Relu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd2
lstm_cell_71/mul_1
lstm_cell_71/add_1AddV2lstm_cell_71/mul:z:0lstm_cell_71/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd2
lstm_cell_71/add_1
lstm_cell_71/Sigmoid_2Sigmoidlstm_cell_71/split:output:3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd2
lstm_cell_71/Sigmoid_2|
lstm_cell_71/Relu_1Relulstm_cell_71/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd2
lstm_cell_71/Relu_1 
lstm_cell_71/mul_2Mullstm_cell_71/Sigmoid_2:y:0!lstm_cell_71/Relu_1:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd2
lstm_cell_71/mul_2
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿd   2
TensorArrayV2_1/element_shape¸
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2_1N
timeConst*
_output_shapes
: *
dtype0*
value	B : 2
time
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counterî
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0+lstm_cell_71_matmul_readvariableop_resource-lstm_cell_71_matmul_1_readvariableop_resource,lstm_cell_71_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd: : : : : *%
_read_only_resource_inputs
	
*
bodyR
while_body_625272*
condR
while_cond_625271*K
output_shapes:
8: : : : :ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd: : : : : *
parallel_iterations 2
whileµ
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿd   22
0TensorArrayV2Stack/TensorListStack/element_shapeè
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿd*
element_dtype02$
"TensorArrayV2Stack/TensorListStack
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿ2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd*
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/perm¥
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿd2
transpose_1f
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2	
runtimeæ
IdentityIdentitystrided_slice_3:output:0$^lstm_cell_71/BiasAdd/ReadVariableOp#^lstm_cell_71/MatMul/ReadVariableOp%^lstm_cell_71/MatMul_1/ReadVariableOp^while*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:ÿÿÿÿÿÿÿÿÿ:::2J
#lstm_cell_71/BiasAdd/ReadVariableOp#lstm_cell_71/BiasAdd/ReadVariableOp2H
"lstm_cell_71/MatMul/ReadVariableOp"lstm_cell_71/MatMul/ReadVariableOp2L
$lstm_cell_71/MatMul_1/ReadVariableOp$lstm_cell_71/MatMul_1/ReadVariableOp2
whilewhile:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
%

while_body_623695
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0
while_lstm_cell_71_623719_0
while_lstm_cell_71_623721_0
while_lstm_cell_71_623723_0
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor
while_lstm_cell_71_623719
while_lstm_cell_71_623721
while_lstm_cell_71_623723¢*while/lstm_cell_71/StatefulPartitionedCallÃ
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   29
7while/TensorArrayV2Read/TensorListGetItem/element_shapeÓ
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItemá
*while/lstm_cell_71/StatefulPartitionedCallStatefulPartitionedCall0while/TensorArrayV2Read/TensorListGetItem:item:0while_placeholder_2while_placeholder_3while_lstm_cell_71_623719_0while_lstm_cell_71_623721_0while_lstm_cell_71_623723_0*
Tin

2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *Q
fLRJ
H__inference_lstm_cell_71_layer_call_and_return_conditional_losses_6233682,
*while/lstm_cell_71/StatefulPartitionedCall÷
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder3while/lstm_cell_71/StatefulPartitionedCall:output:0*
_output_shapes
: *
element_dtype02,
*while/TensorArrayV2Write/TensorListSetItem\
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add/yi
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: 2
	while/add`
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add_1/yv
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: 2
while/add_1
while/IdentityIdentitywhile/add_1:z:0+^while/lstm_cell_71/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity
while/Identity_1Identitywhile_while_maximum_iterations+^while/lstm_cell_71/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity_1
while/Identity_2Identitywhile/add:z:0+^while/lstm_cell_71/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity_2º
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0+^while/lstm_cell_71/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity_3Ä
while/Identity_4Identity3while/lstm_cell_71/StatefulPartitionedCall:output:1+^while/lstm_cell_71/StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd2
while/Identity_4Ä
while/Identity_5Identity3while/lstm_cell_71/StatefulPartitionedCall:output:2+^while/lstm_cell_71/StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd2
while/Identity_5")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"8
while_lstm_cell_71_623719while_lstm_cell_71_623719_0"8
while_lstm_cell_71_623721while_lstm_cell_71_623721_0"8
while_lstm_cell_71_623723while_lstm_cell_71_623723_0"0
while_strided_slice_1while_strided_slice_1_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*Q
_input_shapes@
>: : : : :ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd: : :::2X
*while/lstm_cell_71/StatefulPartitionedCall*while/lstm_cell_71/StatefulPartitionedCall: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd:-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd:

_output_shapes
: :

_output_shapes
: 
³
Ý
H__inference_lstm_cell_71_layer_call_and_return_conditional_losses_625464

inputs
states_0
states_1"
matmul_readvariableop_resource$
 matmul_1_readvariableop_resource#
biasadd_readvariableop_resource
identity

identity_1

identity_2¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp¢MatMul_1/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
MatMul
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource*
_output_shapes
:	d*
dtype02
MatMul_1/ReadVariableOp|
MatMul_1MatMulstates_0MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

MatMul_1l
addAddV2MatMul:product:0MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
add
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype02
BiasAdd/ReadVariableOpy
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
BiasAddP
ConstConst*
_output_shapes
: *
dtype0*
value	B :2
Constd
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split/split_dim¿
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd*
	num_split2
split_
SigmoidSigmoidsplit:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd2	
Sigmoidc
	Sigmoid_1Sigmoidsplit:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd2
	Sigmoid_1\
mulMulSigmoid_1:y:0states_1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd2
mulV
ReluRelusplit:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd2
Reluh
mul_1MulSigmoid:y:0Relu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd2
mul_1]
add_1AddV2mul:z:0	mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd2
add_1c
	Sigmoid_2Sigmoidsplit:output:3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd2
	Sigmoid_2U
Relu_1Relu	add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd2
Relu_1l
mul_2MulSigmoid_2:y:0Relu_1:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd2
mul_2¨
IdentityIdentity	mul_2:z:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd2

Identity¬

Identity_1Identity	mul_2:z:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd2

Identity_1¬

Identity_2Identity	add_1:z:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd2

Identity_2"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*X
_input_shapesG
E:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd:::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs:QM
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
"
_user_specified_name
states/0:QM
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
"
_user_specified_name
states/1
ó6
¬

__inference__traced_save_625587
file_prefix.
*savev2_dense_71_kernel_read_readvariableop,
(savev2_dense_71_bias_read_readvariableop(
$savev2_adam_iter_read_readvariableop	*
&savev2_adam_beta_1_read_readvariableop*
&savev2_adam_beta_2_read_readvariableop)
%savev2_adam_decay_read_readvariableop1
-savev2_adam_learning_rate_read_readvariableop:
6savev2_lstm_71_lstm_cell_71_kernel_read_readvariableopD
@savev2_lstm_71_lstm_cell_71_recurrent_kernel_read_readvariableop8
4savev2_lstm_71_lstm_cell_71_bias_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop5
1savev2_adam_dense_71_kernel_m_read_readvariableop3
/savev2_adam_dense_71_bias_m_read_readvariableopA
=savev2_adam_lstm_71_lstm_cell_71_kernel_m_read_readvariableopK
Gsavev2_adam_lstm_71_lstm_cell_71_recurrent_kernel_m_read_readvariableop?
;savev2_adam_lstm_71_lstm_cell_71_bias_m_read_readvariableop5
1savev2_adam_dense_71_kernel_v_read_readvariableop3
/savev2_adam_dense_71_bias_v_read_readvariableopA
=savev2_adam_lstm_71_lstm_cell_71_kernel_v_read_readvariableopK
Gsavev2_adam_lstm_71_lstm_cell_71_recurrent_kernel_v_read_readvariableop?
;savev2_adam_lstm_71_lstm_cell_71_bias_v_read_readvariableop
savev2_const

identity_1¢MergeV2Checkpoints
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*2
StaticRegexFullMatchc
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.part2
Constl
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/part2	
Const_1
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: 2
Selectt

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: 2

StringJoinZ

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :2

num_shards
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : 2
ShardedFilename/shard¦
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilename¼
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*Î

valueÄ
BÁ
B6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_names¶
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*A
value8B6B B B B B B B B B B B B B B B B B B B B B B B 2
SaveV2/shape_and_slices¸

SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0*savev2_dense_71_kernel_read_readvariableop(savev2_dense_71_bias_read_readvariableop$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop-savev2_adam_learning_rate_read_readvariableop6savev2_lstm_71_lstm_cell_71_kernel_read_readvariableop@savev2_lstm_71_lstm_cell_71_recurrent_kernel_read_readvariableop4savev2_lstm_71_lstm_cell_71_bias_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop1savev2_adam_dense_71_kernel_m_read_readvariableop/savev2_adam_dense_71_bias_m_read_readvariableop=savev2_adam_lstm_71_lstm_cell_71_kernel_m_read_readvariableopGsavev2_adam_lstm_71_lstm_cell_71_recurrent_kernel_m_read_readvariableop;savev2_adam_lstm_71_lstm_cell_71_bias_m_read_readvariableop1savev2_adam_dense_71_kernel_v_read_readvariableop/savev2_adam_dense_71_bias_v_read_readvariableop=savev2_adam_lstm_71_lstm_cell_71_kernel_v_read_readvariableopGsavev2_adam_lstm_71_lstm_cell_71_recurrent_kernel_v_read_readvariableop;savev2_adam_lstm_71_lstm_cell_71_bias_v_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *%
dtypes
2	2
SaveV2º
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:2(
&MergeV2Checkpoints/checkpoint_prefixes¡
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 2
MergeV2Checkpointsr
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: 2

Identitym

Identity_1IdentityIdentity:output:0^MergeV2Checkpoints*
T0*
_output_shapes
: 2

Identity_1"!

identity_1Identity_1:output:0*®
_input_shapes
: :d:: : : : : :	:	d:: : :d::	:	d::d::	:	d:: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:$ 

_output_shapes

:d: 

_output_shapes
::

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :%!

_output_shapes
:	:%	!

_output_shapes
:	d:!


_output_shapes	
::

_output_shapes
: :

_output_shapes
: :$ 

_output_shapes

:d: 

_output_shapes
::%!

_output_shapes
:	:%!

_output_shapes
:	d:!

_output_shapes	
::$ 

_output_shapes

:d: 

_output_shapes
::%!

_output_shapes
:	:%!

_output_shapes
:	d:!

_output_shapes	
::

_output_shapes
: 
C
þ
while_body_625119
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_07
3while_lstm_cell_71_matmul_readvariableop_resource_09
5while_lstm_cell_71_matmul_1_readvariableop_resource_08
4while_lstm_cell_71_biasadd_readvariableop_resource_0
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor5
1while_lstm_cell_71_matmul_readvariableop_resource7
3while_lstm_cell_71_matmul_1_readvariableop_resource6
2while_lstm_cell_71_biasadd_readvariableop_resource¢)while/lstm_cell_71/BiasAdd/ReadVariableOp¢(while/lstm_cell_71/MatMul/ReadVariableOp¢*while/lstm_cell_71/MatMul_1/ReadVariableOpÃ
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   29
7while/TensorArrayV2Read/TensorListGetItem/element_shapeÓ
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItemÉ
(while/lstm_cell_71/MatMul/ReadVariableOpReadVariableOp3while_lstm_cell_71_matmul_readvariableop_resource_0*
_output_shapes
:	*
dtype02*
(while/lstm_cell_71/MatMul/ReadVariableOp×
while/lstm_cell_71/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:00while/lstm_cell_71/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/lstm_cell_71/MatMulÏ
*while/lstm_cell_71/MatMul_1/ReadVariableOpReadVariableOp5while_lstm_cell_71_matmul_1_readvariableop_resource_0*
_output_shapes
:	d*
dtype02,
*while/lstm_cell_71/MatMul_1/ReadVariableOpÀ
while/lstm_cell_71/MatMul_1MatMulwhile_placeholder_22while/lstm_cell_71/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/lstm_cell_71/MatMul_1¸
while/lstm_cell_71/addAddV2#while/lstm_cell_71/MatMul:product:0%while/lstm_cell_71/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/lstm_cell_71/addÈ
)while/lstm_cell_71/BiasAdd/ReadVariableOpReadVariableOp4while_lstm_cell_71_biasadd_readvariableop_resource_0*
_output_shapes	
:*
dtype02+
)while/lstm_cell_71/BiasAdd/ReadVariableOpÅ
while/lstm_cell_71/BiasAddBiasAddwhile/lstm_cell_71/add:z:01while/lstm_cell_71/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/lstm_cell_71/BiasAddv
while/lstm_cell_71/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
while/lstm_cell_71/Const
"while/lstm_cell_71/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2$
"while/lstm_cell_71/split/split_dim
while/lstm_cell_71/splitSplit+while/lstm_cell_71/split/split_dim:output:0#while/lstm_cell_71/BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd*
	num_split2
while/lstm_cell_71/split
while/lstm_cell_71/SigmoidSigmoid!while/lstm_cell_71/split:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd2
while/lstm_cell_71/Sigmoid
while/lstm_cell_71/Sigmoid_1Sigmoid!while/lstm_cell_71/split:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd2
while/lstm_cell_71/Sigmoid_1 
while/lstm_cell_71/mulMul while/lstm_cell_71/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd2
while/lstm_cell_71/mul
while/lstm_cell_71/ReluRelu!while/lstm_cell_71/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd2
while/lstm_cell_71/Relu´
while/lstm_cell_71/mul_1Mulwhile/lstm_cell_71/Sigmoid:y:0%while/lstm_cell_71/Relu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd2
while/lstm_cell_71/mul_1©
while/lstm_cell_71/add_1AddV2while/lstm_cell_71/mul:z:0while/lstm_cell_71/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd2
while/lstm_cell_71/add_1
while/lstm_cell_71/Sigmoid_2Sigmoid!while/lstm_cell_71/split:output:3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd2
while/lstm_cell_71/Sigmoid_2
while/lstm_cell_71/Relu_1Reluwhile/lstm_cell_71/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd2
while/lstm_cell_71/Relu_1¸
while/lstm_cell_71/mul_2Mul while/lstm_cell_71/Sigmoid_2:y:0'while/lstm_cell_71/Relu_1:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd2
while/lstm_cell_71/mul_2à
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_71/mul_2:z:0*
_output_shapes
: *
element_dtype02,
*while/TensorArrayV2Write/TensorListSetItem\
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add/yi
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: 2
	while/add`
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add_1/yv
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: 2
while/add_1â
while/IdentityIdentitywhile/add_1:z:0*^while/lstm_cell_71/BiasAdd/ReadVariableOp)^while/lstm_cell_71/MatMul/ReadVariableOp+^while/lstm_cell_71/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identityõ
while/Identity_1Identitywhile_while_maximum_iterations*^while/lstm_cell_71/BiasAdd/ReadVariableOp)^while/lstm_cell_71/MatMul/ReadVariableOp+^while/lstm_cell_71/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_1ä
while/Identity_2Identitywhile/add:z:0*^while/lstm_cell_71/BiasAdd/ReadVariableOp)^while/lstm_cell_71/MatMul/ReadVariableOp+^while/lstm_cell_71/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_2
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0*^while/lstm_cell_71/BiasAdd/ReadVariableOp)^while/lstm_cell_71/MatMul/ReadVariableOp+^while/lstm_cell_71/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_3
while/Identity_4Identitywhile/lstm_cell_71/mul_2:z:0*^while/lstm_cell_71/BiasAdd/ReadVariableOp)^while/lstm_cell_71/MatMul/ReadVariableOp+^while/lstm_cell_71/MatMul_1/ReadVariableOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd2
while/Identity_4
while/Identity_5Identitywhile/lstm_cell_71/add_1:z:0*^while/lstm_cell_71/BiasAdd/ReadVariableOp)^while/lstm_cell_71/MatMul/ReadVariableOp+^while/lstm_cell_71/MatMul_1/ReadVariableOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd2
while/Identity_5")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"j
2while_lstm_cell_71_biasadd_readvariableop_resource4while_lstm_cell_71_biasadd_readvariableop_resource_0"l
3while_lstm_cell_71_matmul_1_readvariableop_resource5while_lstm_cell_71_matmul_1_readvariableop_resource_0"h
1while_lstm_cell_71_matmul_readvariableop_resource3while_lstm_cell_71_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*Q
_input_shapes@
>: : : : :ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd: : :::2V
)while/lstm_cell_71/BiasAdd/ReadVariableOp)while/lstm_cell_71/BiasAdd/ReadVariableOp2T
(while/lstm_cell_71/MatMul/ReadVariableOp(while/lstm_cell_71/MatMul/ReadVariableOp2X
*while/lstm_cell_71/MatMul_1/ReadVariableOp*while/lstm_cell_71/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd:-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd:

_output_shapes
: :

_output_shapes
: 
	
Ý
D__inference_dense_71_layer_call_and_return_conditional_losses_625389

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:d*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
BiasAdd
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿd::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
 
_user_specified_nameinputs
	
Ý
D__inference_dense_71_layer_call_and_return_conditional_losses_624254

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:d*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
BiasAdd
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿd::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
 
_user_specified_nameinputs
C
þ
while_body_624129
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_07
3while_lstm_cell_71_matmul_readvariableop_resource_09
5while_lstm_cell_71_matmul_1_readvariableop_resource_08
4while_lstm_cell_71_biasadd_readvariableop_resource_0
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor5
1while_lstm_cell_71_matmul_readvariableop_resource7
3while_lstm_cell_71_matmul_1_readvariableop_resource6
2while_lstm_cell_71_biasadd_readvariableop_resource¢)while/lstm_cell_71/BiasAdd/ReadVariableOp¢(while/lstm_cell_71/MatMul/ReadVariableOp¢*while/lstm_cell_71/MatMul_1/ReadVariableOpÃ
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   29
7while/TensorArrayV2Read/TensorListGetItem/element_shapeÓ
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItemÉ
(while/lstm_cell_71/MatMul/ReadVariableOpReadVariableOp3while_lstm_cell_71_matmul_readvariableop_resource_0*
_output_shapes
:	*
dtype02*
(while/lstm_cell_71/MatMul/ReadVariableOp×
while/lstm_cell_71/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:00while/lstm_cell_71/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/lstm_cell_71/MatMulÏ
*while/lstm_cell_71/MatMul_1/ReadVariableOpReadVariableOp5while_lstm_cell_71_matmul_1_readvariableop_resource_0*
_output_shapes
:	d*
dtype02,
*while/lstm_cell_71/MatMul_1/ReadVariableOpÀ
while/lstm_cell_71/MatMul_1MatMulwhile_placeholder_22while/lstm_cell_71/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/lstm_cell_71/MatMul_1¸
while/lstm_cell_71/addAddV2#while/lstm_cell_71/MatMul:product:0%while/lstm_cell_71/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/lstm_cell_71/addÈ
)while/lstm_cell_71/BiasAdd/ReadVariableOpReadVariableOp4while_lstm_cell_71_biasadd_readvariableop_resource_0*
_output_shapes	
:*
dtype02+
)while/lstm_cell_71/BiasAdd/ReadVariableOpÅ
while/lstm_cell_71/BiasAddBiasAddwhile/lstm_cell_71/add:z:01while/lstm_cell_71/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/lstm_cell_71/BiasAddv
while/lstm_cell_71/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
while/lstm_cell_71/Const
"while/lstm_cell_71/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2$
"while/lstm_cell_71/split/split_dim
while/lstm_cell_71/splitSplit+while/lstm_cell_71/split/split_dim:output:0#while/lstm_cell_71/BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd*
	num_split2
while/lstm_cell_71/split
while/lstm_cell_71/SigmoidSigmoid!while/lstm_cell_71/split:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd2
while/lstm_cell_71/Sigmoid
while/lstm_cell_71/Sigmoid_1Sigmoid!while/lstm_cell_71/split:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd2
while/lstm_cell_71/Sigmoid_1 
while/lstm_cell_71/mulMul while/lstm_cell_71/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd2
while/lstm_cell_71/mul
while/lstm_cell_71/ReluRelu!while/lstm_cell_71/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd2
while/lstm_cell_71/Relu´
while/lstm_cell_71/mul_1Mulwhile/lstm_cell_71/Sigmoid:y:0%while/lstm_cell_71/Relu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd2
while/lstm_cell_71/mul_1©
while/lstm_cell_71/add_1AddV2while/lstm_cell_71/mul:z:0while/lstm_cell_71/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd2
while/lstm_cell_71/add_1
while/lstm_cell_71/Sigmoid_2Sigmoid!while/lstm_cell_71/split:output:3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd2
while/lstm_cell_71/Sigmoid_2
while/lstm_cell_71/Relu_1Reluwhile/lstm_cell_71/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd2
while/lstm_cell_71/Relu_1¸
while/lstm_cell_71/mul_2Mul while/lstm_cell_71/Sigmoid_2:y:0'while/lstm_cell_71/Relu_1:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd2
while/lstm_cell_71/mul_2à
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_71/mul_2:z:0*
_output_shapes
: *
element_dtype02,
*while/TensorArrayV2Write/TensorListSetItem\
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add/yi
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: 2
	while/add`
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add_1/yv
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: 2
while/add_1â
while/IdentityIdentitywhile/add_1:z:0*^while/lstm_cell_71/BiasAdd/ReadVariableOp)^while/lstm_cell_71/MatMul/ReadVariableOp+^while/lstm_cell_71/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identityõ
while/Identity_1Identitywhile_while_maximum_iterations*^while/lstm_cell_71/BiasAdd/ReadVariableOp)^while/lstm_cell_71/MatMul/ReadVariableOp+^while/lstm_cell_71/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_1ä
while/Identity_2Identitywhile/add:z:0*^while/lstm_cell_71/BiasAdd/ReadVariableOp)^while/lstm_cell_71/MatMul/ReadVariableOp+^while/lstm_cell_71/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_2
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0*^while/lstm_cell_71/BiasAdd/ReadVariableOp)^while/lstm_cell_71/MatMul/ReadVariableOp+^while/lstm_cell_71/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_3
while/Identity_4Identitywhile/lstm_cell_71/mul_2:z:0*^while/lstm_cell_71/BiasAdd/ReadVariableOp)^while/lstm_cell_71/MatMul/ReadVariableOp+^while/lstm_cell_71/MatMul_1/ReadVariableOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd2
while/Identity_4
while/Identity_5Identitywhile/lstm_cell_71/add_1:z:0*^while/lstm_cell_71/BiasAdd/ReadVariableOp)^while/lstm_cell_71/MatMul/ReadVariableOp+^while/lstm_cell_71/MatMul_1/ReadVariableOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd2
while/Identity_5")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"j
2while_lstm_cell_71_biasadd_readvariableop_resource4while_lstm_cell_71_biasadd_readvariableop_resource_0"l
3while_lstm_cell_71_matmul_1_readvariableop_resource5while_lstm_cell_71_matmul_1_readvariableop_resource_0"h
1while_lstm_cell_71_matmul_readvariableop_resource3while_lstm_cell_71_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*Q
_input_shapes@
>: : : : :ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd: : :::2V
)while/lstm_cell_71/BiasAdd/ReadVariableOp)while/lstm_cell_71/BiasAdd/ReadVariableOp2T
(while/lstm_cell_71/MatMul/ReadVariableOp(while/lstm_cell_71/MatMul/ReadVariableOp2X
*while/lstm_cell_71/MatMul_1/ReadVariableOp*while/lstm_cell_71/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd:-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd:

_output_shapes
: :

_output_shapes
: 
«
Ã
while_cond_623826
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_14
0while_while_cond_623826___redundant_placeholder04
0while_while_cond_623826___redundant_placeholder14
0while_while_cond_623826___redundant_placeholder24
0while_while_cond_623826___redundant_placeholder3
while_identity
p

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: 2

while/Less]
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: 2
while/Identity")
while_identitywhile/Identity:output:0*S
_input_shapesB
@: : : : :ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd:-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd:

_output_shapes
: :

_output_shapes
:
ä
·
.__inference_sequential_71_layer_call_fn_624350
lstm_71_input
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
identity¢StatefulPartitionedCall§
StatefulPartitionedCallStatefulPartitionedCalllstm_71_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3*
Tin

2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*'
_read_only_resource_inputs	
*-
config_proto

CPU

GPU 2J 8 *R
fMRK
I__inference_sequential_71_layer_call_and_return_conditional_losses_6243372
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:ÿÿÿÿÿÿÿÿÿ:::::22
StatefulPartitionedCallStatefulPartitionedCall:Z V
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
'
_user_specified_namelstm_71_input


(__inference_lstm_71_layer_call_fn_625051
inputs_0
unknown
	unknown_0
	unknown_1
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputs_0unknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_lstm_71_layer_call_and_return_conditional_losses_6238962
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:::22
StatefulPartitionedCallStatefulPartitionedCall:^ Z
4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
inputs/0
C
þ
while_body_625272
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_07
3while_lstm_cell_71_matmul_readvariableop_resource_09
5while_lstm_cell_71_matmul_1_readvariableop_resource_08
4while_lstm_cell_71_biasadd_readvariableop_resource_0
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor5
1while_lstm_cell_71_matmul_readvariableop_resource7
3while_lstm_cell_71_matmul_1_readvariableop_resource6
2while_lstm_cell_71_biasadd_readvariableop_resource¢)while/lstm_cell_71/BiasAdd/ReadVariableOp¢(while/lstm_cell_71/MatMul/ReadVariableOp¢*while/lstm_cell_71/MatMul_1/ReadVariableOpÃ
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   29
7while/TensorArrayV2Read/TensorListGetItem/element_shapeÓ
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItemÉ
(while/lstm_cell_71/MatMul/ReadVariableOpReadVariableOp3while_lstm_cell_71_matmul_readvariableop_resource_0*
_output_shapes
:	*
dtype02*
(while/lstm_cell_71/MatMul/ReadVariableOp×
while/lstm_cell_71/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:00while/lstm_cell_71/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/lstm_cell_71/MatMulÏ
*while/lstm_cell_71/MatMul_1/ReadVariableOpReadVariableOp5while_lstm_cell_71_matmul_1_readvariableop_resource_0*
_output_shapes
:	d*
dtype02,
*while/lstm_cell_71/MatMul_1/ReadVariableOpÀ
while/lstm_cell_71/MatMul_1MatMulwhile_placeholder_22while/lstm_cell_71/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/lstm_cell_71/MatMul_1¸
while/lstm_cell_71/addAddV2#while/lstm_cell_71/MatMul:product:0%while/lstm_cell_71/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/lstm_cell_71/addÈ
)while/lstm_cell_71/BiasAdd/ReadVariableOpReadVariableOp4while_lstm_cell_71_biasadd_readvariableop_resource_0*
_output_shapes	
:*
dtype02+
)while/lstm_cell_71/BiasAdd/ReadVariableOpÅ
while/lstm_cell_71/BiasAddBiasAddwhile/lstm_cell_71/add:z:01while/lstm_cell_71/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/lstm_cell_71/BiasAddv
while/lstm_cell_71/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
while/lstm_cell_71/Const
"while/lstm_cell_71/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2$
"while/lstm_cell_71/split/split_dim
while/lstm_cell_71/splitSplit+while/lstm_cell_71/split/split_dim:output:0#while/lstm_cell_71/BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd*
	num_split2
while/lstm_cell_71/split
while/lstm_cell_71/SigmoidSigmoid!while/lstm_cell_71/split:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd2
while/lstm_cell_71/Sigmoid
while/lstm_cell_71/Sigmoid_1Sigmoid!while/lstm_cell_71/split:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd2
while/lstm_cell_71/Sigmoid_1 
while/lstm_cell_71/mulMul while/lstm_cell_71/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd2
while/lstm_cell_71/mul
while/lstm_cell_71/ReluRelu!while/lstm_cell_71/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd2
while/lstm_cell_71/Relu´
while/lstm_cell_71/mul_1Mulwhile/lstm_cell_71/Sigmoid:y:0%while/lstm_cell_71/Relu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd2
while/lstm_cell_71/mul_1©
while/lstm_cell_71/add_1AddV2while/lstm_cell_71/mul:z:0while/lstm_cell_71/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd2
while/lstm_cell_71/add_1
while/lstm_cell_71/Sigmoid_2Sigmoid!while/lstm_cell_71/split:output:3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd2
while/lstm_cell_71/Sigmoid_2
while/lstm_cell_71/Relu_1Reluwhile/lstm_cell_71/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd2
while/lstm_cell_71/Relu_1¸
while/lstm_cell_71/mul_2Mul while/lstm_cell_71/Sigmoid_2:y:0'while/lstm_cell_71/Relu_1:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd2
while/lstm_cell_71/mul_2à
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_71/mul_2:z:0*
_output_shapes
: *
element_dtype02,
*while/TensorArrayV2Write/TensorListSetItem\
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add/yi
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: 2
	while/add`
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add_1/yv
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: 2
while/add_1â
while/IdentityIdentitywhile/add_1:z:0*^while/lstm_cell_71/BiasAdd/ReadVariableOp)^while/lstm_cell_71/MatMul/ReadVariableOp+^while/lstm_cell_71/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identityõ
while/Identity_1Identitywhile_while_maximum_iterations*^while/lstm_cell_71/BiasAdd/ReadVariableOp)^while/lstm_cell_71/MatMul/ReadVariableOp+^while/lstm_cell_71/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_1ä
while/Identity_2Identitywhile/add:z:0*^while/lstm_cell_71/BiasAdd/ReadVariableOp)^while/lstm_cell_71/MatMul/ReadVariableOp+^while/lstm_cell_71/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_2
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0*^while/lstm_cell_71/BiasAdd/ReadVariableOp)^while/lstm_cell_71/MatMul/ReadVariableOp+^while/lstm_cell_71/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_3
while/Identity_4Identitywhile/lstm_cell_71/mul_2:z:0*^while/lstm_cell_71/BiasAdd/ReadVariableOp)^while/lstm_cell_71/MatMul/ReadVariableOp+^while/lstm_cell_71/MatMul_1/ReadVariableOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd2
while/Identity_4
while/Identity_5Identitywhile/lstm_cell_71/add_1:z:0*^while/lstm_cell_71/BiasAdd/ReadVariableOp)^while/lstm_cell_71/MatMul/ReadVariableOp+^while/lstm_cell_71/MatMul_1/ReadVariableOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd2
while/Identity_5")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"j
2while_lstm_cell_71_biasadd_readvariableop_resource4while_lstm_cell_71_biasadd_readvariableop_resource_0"l
3while_lstm_cell_71_matmul_1_readvariableop_resource5while_lstm_cell_71_matmul_1_readvariableop_resource_0"h
1while_lstm_cell_71_matmul_readvariableop_resource3while_lstm_cell_71_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*Q
_input_shapes@
>: : : : :ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd: : :::2V
)while/lstm_cell_71/BiasAdd/ReadVariableOp)while/lstm_cell_71/BiasAdd/ReadVariableOp2T
(while/lstm_cell_71/MatMul/ReadVariableOp(while/lstm_cell_71/MatMul/ReadVariableOp2X
*while/lstm_cell_71/MatMul_1/ReadVariableOp*while/lstm_cell_71/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd:-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd:

_output_shapes
: :

_output_shapes
: 
×r
Î
I__inference_sequential_71_layer_call_and_return_conditional_losses_624534

inputs7
3lstm_71_lstm_cell_71_matmul_readvariableop_resource9
5lstm_71_lstm_cell_71_matmul_1_readvariableop_resource8
4lstm_71_lstm_cell_71_biasadd_readvariableop_resource+
'dense_71_matmul_readvariableop_resource,
(dense_71_biasadd_readvariableop_resource
identity¢dense_71/BiasAdd/ReadVariableOp¢dense_71/MatMul/ReadVariableOp¢+lstm_71/lstm_cell_71/BiasAdd/ReadVariableOp¢*lstm_71/lstm_cell_71/MatMul/ReadVariableOp¢,lstm_71/lstm_cell_71/MatMul_1/ReadVariableOp¢lstm_71/whileT
lstm_71/ShapeShapeinputs*
T0*
_output_shapes
:2
lstm_71/Shape
lstm_71/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
lstm_71/strided_slice/stack
lstm_71/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
lstm_71/strided_slice/stack_1
lstm_71/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
lstm_71/strided_slice/stack_2
lstm_71/strided_sliceStridedSlicelstm_71/Shape:output:0$lstm_71/strided_slice/stack:output:0&lstm_71/strided_slice/stack_1:output:0&lstm_71/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
lstm_71/strided_slicel
lstm_71/zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :d2
lstm_71/zeros/mul/y
lstm_71/zeros/mulMullstm_71/strided_slice:output:0lstm_71/zeros/mul/y:output:0*
T0*
_output_shapes
: 2
lstm_71/zeros/mulo
lstm_71/zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :è2
lstm_71/zeros/Less/y
lstm_71/zeros/LessLesslstm_71/zeros/mul:z:0lstm_71/zeros/Less/y:output:0*
T0*
_output_shapes
: 2
lstm_71/zeros/Lessr
lstm_71/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :d2
lstm_71/zeros/packed/1£
lstm_71/zeros/packedPacklstm_71/strided_slice:output:0lstm_71/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
lstm_71/zeros/packedo
lstm_71/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
lstm_71/zeros/Const
lstm_71/zerosFilllstm_71/zeros/packed:output:0lstm_71/zeros/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd2
lstm_71/zerosp
lstm_71/zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B :d2
lstm_71/zeros_1/mul/y
lstm_71/zeros_1/mulMullstm_71/strided_slice:output:0lstm_71/zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
lstm_71/zeros_1/muls
lstm_71/zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :è2
lstm_71/zeros_1/Less/y
lstm_71/zeros_1/LessLesslstm_71/zeros_1/mul:z:0lstm_71/zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
lstm_71/zeros_1/Lessv
lstm_71/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :d2
lstm_71/zeros_1/packed/1©
lstm_71/zeros_1/packedPacklstm_71/strided_slice:output:0!lstm_71/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
lstm_71/zeros_1/packeds
lstm_71/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
lstm_71/zeros_1/Const
lstm_71/zeros_1Filllstm_71/zeros_1/packed:output:0lstm_71/zeros_1/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd2
lstm_71/zeros_1
lstm_71/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
lstm_71/transpose/perm
lstm_71/transpose	Transposeinputslstm_71/transpose/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_71/transposeg
lstm_71/Shape_1Shapelstm_71/transpose:y:0*
T0*
_output_shapes
:2
lstm_71/Shape_1
lstm_71/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
lstm_71/strided_slice_1/stack
lstm_71/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2!
lstm_71/strided_slice_1/stack_1
lstm_71/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2!
lstm_71/strided_slice_1/stack_2
lstm_71/strided_slice_1StridedSlicelstm_71/Shape_1:output:0&lstm_71/strided_slice_1/stack:output:0(lstm_71/strided_slice_1/stack_1:output:0(lstm_71/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
lstm_71/strided_slice_1
#lstm_71/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2%
#lstm_71/TensorArrayV2/element_shapeÒ
lstm_71/TensorArrayV2TensorListReserve,lstm_71/TensorArrayV2/element_shape:output:0 lstm_71/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
lstm_71/TensorArrayV2Ï
=lstm_71/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   2?
=lstm_71/TensorArrayUnstack/TensorListFromTensor/element_shape
/lstm_71/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorlstm_71/transpose:y:0Flstm_71/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type021
/lstm_71/TensorArrayUnstack/TensorListFromTensor
lstm_71/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
lstm_71/strided_slice_2/stack
lstm_71/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2!
lstm_71/strided_slice_2/stack_1
lstm_71/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2!
lstm_71/strided_slice_2/stack_2¬
lstm_71/strided_slice_2StridedSlicelstm_71/transpose:y:0&lstm_71/strided_slice_2/stack:output:0(lstm_71/strided_slice_2/stack_1:output:0(lstm_71/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
shrink_axis_mask2
lstm_71/strided_slice_2Í
*lstm_71/lstm_cell_71/MatMul/ReadVariableOpReadVariableOp3lstm_71_lstm_cell_71_matmul_readvariableop_resource*
_output_shapes
:	*
dtype02,
*lstm_71/lstm_cell_71/MatMul/ReadVariableOpÍ
lstm_71/lstm_cell_71/MatMulMatMul lstm_71/strided_slice_2:output:02lstm_71/lstm_cell_71/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_71/lstm_cell_71/MatMulÓ
,lstm_71/lstm_cell_71/MatMul_1/ReadVariableOpReadVariableOp5lstm_71_lstm_cell_71_matmul_1_readvariableop_resource*
_output_shapes
:	d*
dtype02.
,lstm_71/lstm_cell_71/MatMul_1/ReadVariableOpÉ
lstm_71/lstm_cell_71/MatMul_1MatMullstm_71/zeros:output:04lstm_71/lstm_cell_71/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_71/lstm_cell_71/MatMul_1À
lstm_71/lstm_cell_71/addAddV2%lstm_71/lstm_cell_71/MatMul:product:0'lstm_71/lstm_cell_71/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_71/lstm_cell_71/addÌ
+lstm_71/lstm_cell_71/BiasAdd/ReadVariableOpReadVariableOp4lstm_71_lstm_cell_71_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02-
+lstm_71/lstm_cell_71/BiasAdd/ReadVariableOpÍ
lstm_71/lstm_cell_71/BiasAddBiasAddlstm_71/lstm_cell_71/add:z:03lstm_71/lstm_cell_71/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_71/lstm_cell_71/BiasAddz
lstm_71/lstm_cell_71/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_71/lstm_cell_71/Const
$lstm_71/lstm_cell_71/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2&
$lstm_71/lstm_cell_71/split/split_dim
lstm_71/lstm_cell_71/splitSplit-lstm_71/lstm_cell_71/split/split_dim:output:0%lstm_71/lstm_cell_71/BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd*
	num_split2
lstm_71/lstm_cell_71/split
lstm_71/lstm_cell_71/SigmoidSigmoid#lstm_71/lstm_cell_71/split:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd2
lstm_71/lstm_cell_71/Sigmoid¢
lstm_71/lstm_cell_71/Sigmoid_1Sigmoid#lstm_71/lstm_cell_71/split:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd2 
lstm_71/lstm_cell_71/Sigmoid_1«
lstm_71/lstm_cell_71/mulMul"lstm_71/lstm_cell_71/Sigmoid_1:y:0lstm_71/zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd2
lstm_71/lstm_cell_71/mul
lstm_71/lstm_cell_71/ReluRelu#lstm_71/lstm_cell_71/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd2
lstm_71/lstm_cell_71/Relu¼
lstm_71/lstm_cell_71/mul_1Mul lstm_71/lstm_cell_71/Sigmoid:y:0'lstm_71/lstm_cell_71/Relu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd2
lstm_71/lstm_cell_71/mul_1±
lstm_71/lstm_cell_71/add_1AddV2lstm_71/lstm_cell_71/mul:z:0lstm_71/lstm_cell_71/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd2
lstm_71/lstm_cell_71/add_1¢
lstm_71/lstm_cell_71/Sigmoid_2Sigmoid#lstm_71/lstm_cell_71/split:output:3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd2 
lstm_71/lstm_cell_71/Sigmoid_2
lstm_71/lstm_cell_71/Relu_1Relulstm_71/lstm_cell_71/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd2
lstm_71/lstm_cell_71/Relu_1À
lstm_71/lstm_cell_71/mul_2Mul"lstm_71/lstm_cell_71/Sigmoid_2:y:0)lstm_71/lstm_cell_71/Relu_1:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd2
lstm_71/lstm_cell_71/mul_2
%lstm_71/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿd   2'
%lstm_71/TensorArrayV2_1/element_shapeØ
lstm_71/TensorArrayV2_1TensorListReserve.lstm_71/TensorArrayV2_1/element_shape:output:0 lstm_71/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
lstm_71/TensorArrayV2_1^
lstm_71/timeConst*
_output_shapes
: *
dtype0*
value	B : 2
lstm_71/time
 lstm_71/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2"
 lstm_71/while/maximum_iterationsz
lstm_71/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
lstm_71/while/loop_counteræ
lstm_71/whileWhile#lstm_71/while/loop_counter:output:0)lstm_71/while/maximum_iterations:output:0lstm_71/time:output:0 lstm_71/TensorArrayV2_1:handle:0lstm_71/zeros:output:0lstm_71/zeros_1:output:0 lstm_71/strided_slice_1:output:0?lstm_71/TensorArrayUnstack/TensorListFromTensor:output_handle:03lstm_71_lstm_cell_71_matmul_readvariableop_resource5lstm_71_lstm_cell_71_matmul_1_readvariableop_resource4lstm_71_lstm_cell_71_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd: : : : : *%
_read_only_resource_inputs
	
*%
bodyR
lstm_71_while_body_624443*%
condR
lstm_71_while_cond_624442*K
output_shapes:
8: : : : :ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd: : : : : *
parallel_iterations 2
lstm_71/whileÅ
8lstm_71/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿd   2:
8lstm_71/TensorArrayV2Stack/TensorListStack/element_shape
*lstm_71/TensorArrayV2Stack/TensorListStackTensorListStacklstm_71/while:output:3Alstm_71/TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿd*
element_dtype02,
*lstm_71/TensorArrayV2Stack/TensorListStack
lstm_71/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿ2
lstm_71/strided_slice_3/stack
lstm_71/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2!
lstm_71/strided_slice_3/stack_1
lstm_71/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2!
lstm_71/strided_slice_3/stack_2Ê
lstm_71/strided_slice_3StridedSlice3lstm_71/TensorArrayV2Stack/TensorListStack:tensor:0&lstm_71/strided_slice_3/stack:output:0(lstm_71/strided_slice_3/stack_1:output:0(lstm_71/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd*
shrink_axis_mask2
lstm_71/strided_slice_3
lstm_71/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
lstm_71/transpose_1/permÅ
lstm_71/transpose_1	Transpose3lstm_71/TensorArrayV2Stack/TensorListStack:tensor:0!lstm_71/transpose_1/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿd2
lstm_71/transpose_1v
lstm_71/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2
lstm_71/runtime¨
dense_71/MatMul/ReadVariableOpReadVariableOp'dense_71_matmul_readvariableop_resource*
_output_shapes

:d*
dtype02 
dense_71/MatMul/ReadVariableOp¨
dense_71/MatMulMatMul lstm_71/strided_slice_3:output:0&dense_71/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_71/MatMul§
dense_71/BiasAdd/ReadVariableOpReadVariableOp(dense_71_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
dense_71/BiasAdd/ReadVariableOp¥
dense_71/BiasAddBiasAdddense_71/MatMul:product:0'dense_71/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_71/BiasAddÊ
IdentityIdentitydense_71/BiasAdd:output:0 ^dense_71/BiasAdd/ReadVariableOp^dense_71/MatMul/ReadVariableOp,^lstm_71/lstm_cell_71/BiasAdd/ReadVariableOp+^lstm_71/lstm_cell_71/MatMul/ReadVariableOp-^lstm_71/lstm_cell_71/MatMul_1/ReadVariableOp^lstm_71/while*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:ÿÿÿÿÿÿÿÿÿ:::::2B
dense_71/BiasAdd/ReadVariableOpdense_71/BiasAdd/ReadVariableOp2@
dense_71/MatMul/ReadVariableOpdense_71/MatMul/ReadVariableOp2Z
+lstm_71/lstm_cell_71/BiasAdd/ReadVariableOp+lstm_71/lstm_cell_71/BiasAdd/ReadVariableOp2X
*lstm_71/lstm_cell_71/MatMul/ReadVariableOp*lstm_71/lstm_cell_71/MatMul/ReadVariableOp2\
,lstm_71/lstm_cell_71/MatMul_1/ReadVariableOp,lstm_71/lstm_cell_71/MatMul_1/ReadVariableOp2
lstm_71/whilelstm_71/while:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Á
û
'sequential_71_lstm_71_while_cond_623203H
Dsequential_71_lstm_71_while_sequential_71_lstm_71_while_loop_counterN
Jsequential_71_lstm_71_while_sequential_71_lstm_71_while_maximum_iterations+
'sequential_71_lstm_71_while_placeholder-
)sequential_71_lstm_71_while_placeholder_1-
)sequential_71_lstm_71_while_placeholder_2-
)sequential_71_lstm_71_while_placeholder_3J
Fsequential_71_lstm_71_while_less_sequential_71_lstm_71_strided_slice_1`
\sequential_71_lstm_71_while_sequential_71_lstm_71_while_cond_623203___redundant_placeholder0`
\sequential_71_lstm_71_while_sequential_71_lstm_71_while_cond_623203___redundant_placeholder1`
\sequential_71_lstm_71_while_sequential_71_lstm_71_while_cond_623203___redundant_placeholder2`
\sequential_71_lstm_71_while_sequential_71_lstm_71_while_cond_623203___redundant_placeholder3(
$sequential_71_lstm_71_while_identity
Þ
 sequential_71/lstm_71/while/LessLess'sequential_71_lstm_71_while_placeholderFsequential_71_lstm_71_while_less_sequential_71_lstm_71_strided_slice_1*
T0*
_output_shapes
: 2"
 sequential_71/lstm_71/while/Less
$sequential_71/lstm_71/while/IdentityIdentity$sequential_71/lstm_71/while/Less:z:0*
T0
*
_output_shapes
: 2&
$sequential_71/lstm_71/while/Identity"U
$sequential_71_lstm_71_while_identity-sequential_71/lstm_71/while/Identity:output:0*S
_input_shapesB
@: : : : :ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd:-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd:

_output_shapes
: :

_output_shapes
:
Ç[
ô
C__inference_lstm_71_layer_call_and_return_conditional_losses_625029
inputs_0/
+lstm_cell_71_matmul_readvariableop_resource1
-lstm_cell_71_matmul_1_readvariableop_resource0
,lstm_cell_71_biasadd_readvariableop_resource
identity¢#lstm_cell_71/BiasAdd/ReadVariableOp¢"lstm_cell_71/MatMul/ReadVariableOp¢$lstm_cell_71/MatMul_1/ReadVariableOp¢whileF
ShapeShapeinputs_0*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2â
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice\
zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :d2
zeros/mul/yl
	zeros/mulMulstrided_slice:output:0zeros/mul/y:output:0*
T0*
_output_shapes
: 2
	zeros/mul_
zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :è2
zeros/Less/yg

zeros/LessLesszeros/mul:z:0zeros/Less/y:output:0*
T0*
_output_shapes
: 2

zeros/Lessb
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :d2
zeros/packed/1
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros/packed_
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros/Constu
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd2
zeros`
zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B :d2
zeros_1/mul/yr
zeros_1/mulMulstrided_slice:output:0zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
zeros_1/mulc
zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :è2
zeros_1/Less/yo
zeros_1/LessLesszeros_1/mul:z:0zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
zeros_1/Lessf
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :d2
zeros_1/packed/1
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros_1/packedc
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros_1/Const}
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd2	
zeros_1u
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/perm
	transpose	Transposeinputs_0transpose/perm:output:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
	transposeO
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:2	
Shape_1x
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2î
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2
TensorArrayV2/element_shape²
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2¿
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   27
5TensorArrayUnstack/TensorListFromTensor/element_shapeø
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02)
'TensorArrayUnstack/TensorListFromTensorx
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2ü
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
shrink_axis_mask2
strided_slice_2µ
"lstm_cell_71/MatMul/ReadVariableOpReadVariableOp+lstm_cell_71_matmul_readvariableop_resource*
_output_shapes
:	*
dtype02$
"lstm_cell_71/MatMul/ReadVariableOp­
lstm_cell_71/MatMulMatMulstrided_slice_2:output:0*lstm_cell_71/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell_71/MatMul»
$lstm_cell_71/MatMul_1/ReadVariableOpReadVariableOp-lstm_cell_71_matmul_1_readvariableop_resource*
_output_shapes
:	d*
dtype02&
$lstm_cell_71/MatMul_1/ReadVariableOp©
lstm_cell_71/MatMul_1MatMulzeros:output:0,lstm_cell_71/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell_71/MatMul_1 
lstm_cell_71/addAddV2lstm_cell_71/MatMul:product:0lstm_cell_71/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell_71/add´
#lstm_cell_71/BiasAdd/ReadVariableOpReadVariableOp,lstm_cell_71_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02%
#lstm_cell_71/BiasAdd/ReadVariableOp­
lstm_cell_71/BiasAddBiasAddlstm_cell_71/add:z:0+lstm_cell_71/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell_71/BiasAddj
lstm_cell_71/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_71/Const~
lstm_cell_71/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_71/split/split_dimó
lstm_cell_71/splitSplit%lstm_cell_71/split/split_dim:output:0lstm_cell_71/BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd*
	num_split2
lstm_cell_71/split
lstm_cell_71/SigmoidSigmoidlstm_cell_71/split:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd2
lstm_cell_71/Sigmoid
lstm_cell_71/Sigmoid_1Sigmoidlstm_cell_71/split:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd2
lstm_cell_71/Sigmoid_1
lstm_cell_71/mulMullstm_cell_71/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd2
lstm_cell_71/mul}
lstm_cell_71/ReluRelulstm_cell_71/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd2
lstm_cell_71/Relu
lstm_cell_71/mul_1Mullstm_cell_71/Sigmoid:y:0lstm_cell_71/Relu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd2
lstm_cell_71/mul_1
lstm_cell_71/add_1AddV2lstm_cell_71/mul:z:0lstm_cell_71/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd2
lstm_cell_71/add_1
lstm_cell_71/Sigmoid_2Sigmoidlstm_cell_71/split:output:3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd2
lstm_cell_71/Sigmoid_2|
lstm_cell_71/Relu_1Relulstm_cell_71/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd2
lstm_cell_71/Relu_1 
lstm_cell_71/mul_2Mullstm_cell_71/Sigmoid_2:y:0!lstm_cell_71/Relu_1:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd2
lstm_cell_71/mul_2
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿd   2
TensorArrayV2_1/element_shape¸
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2_1N
timeConst*
_output_shapes
: *
dtype0*
value	B : 2
time
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counterî
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0+lstm_cell_71_matmul_readvariableop_resource-lstm_cell_71_matmul_1_readvariableop_resource,lstm_cell_71_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd: : : : : *%
_read_only_resource_inputs
	
*
bodyR
while_body_624944*
condR
while_cond_624943*K
output_shapes:
8: : : : :ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd: : : : : *
parallel_iterations 2
whileµ
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿd   22
0TensorArrayV2Stack/TensorListStack/element_shapeñ
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿd*
element_dtype02$
"TensorArrayV2Stack/TensorListStack
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿ2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd*
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/perm®
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿd2
transpose_1f
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2	
runtimeæ
IdentityIdentitystrided_slice_3:output:0$^lstm_cell_71/BiasAdd/ReadVariableOp#^lstm_cell_71/MatMul/ReadVariableOp%^lstm_cell_71/MatMul_1/ReadVariableOp^while*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:::2J
#lstm_cell_71/BiasAdd/ReadVariableOp#lstm_cell_71/BiasAdd/ReadVariableOp2H
"lstm_cell_71/MatMul/ReadVariableOp"lstm_cell_71/MatMul/ReadVariableOp2L
$lstm_cell_71/MatMul_1/ReadVariableOp$lstm_cell_71/MatMul_1/ReadVariableOp2
whilewhile:^ Z
4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
inputs/0
`
»
"__inference__traced_restore_625663
file_prefix$
 assignvariableop_dense_71_kernel$
 assignvariableop_1_dense_71_bias 
assignvariableop_2_adam_iter"
assignvariableop_3_adam_beta_1"
assignvariableop_4_adam_beta_2!
assignvariableop_5_adam_decay)
%assignvariableop_6_adam_learning_rate2
.assignvariableop_7_lstm_71_lstm_cell_71_kernel<
8assignvariableop_8_lstm_71_lstm_cell_71_recurrent_kernel0
,assignvariableop_9_lstm_71_lstm_cell_71_bias
assignvariableop_10_total
assignvariableop_11_count.
*assignvariableop_12_adam_dense_71_kernel_m,
(assignvariableop_13_adam_dense_71_bias_m:
6assignvariableop_14_adam_lstm_71_lstm_cell_71_kernel_mD
@assignvariableop_15_adam_lstm_71_lstm_cell_71_recurrent_kernel_m8
4assignvariableop_16_adam_lstm_71_lstm_cell_71_bias_m.
*assignvariableop_17_adam_dense_71_kernel_v,
(assignvariableop_18_adam_dense_71_bias_v:
6assignvariableop_19_adam_lstm_71_lstm_cell_71_kernel_vD
@assignvariableop_20_adam_lstm_71_lstm_cell_71_recurrent_kernel_v8
4assignvariableop_21_adam_lstm_71_lstm_cell_71_bias_v
identity_23¢AssignVariableOp¢AssignVariableOp_1¢AssignVariableOp_10¢AssignVariableOp_11¢AssignVariableOp_12¢AssignVariableOp_13¢AssignVariableOp_14¢AssignVariableOp_15¢AssignVariableOp_16¢AssignVariableOp_17¢AssignVariableOp_18¢AssignVariableOp_19¢AssignVariableOp_2¢AssignVariableOp_20¢AssignVariableOp_21¢AssignVariableOp_3¢AssignVariableOp_4¢AssignVariableOp_5¢AssignVariableOp_6¢AssignVariableOp_7¢AssignVariableOp_8¢AssignVariableOp_9Â
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*Î

valueÄ
BÁ
B6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2/tensor_names¼
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*A
value8B6B B B B B B B B B B B B B B B B B B B B B B B 2
RestoreV2/shape_and_slices
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*p
_output_shapes^
\:::::::::::::::::::::::*%
dtypes
2	2
	RestoreV2g
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:2

Identity
AssignVariableOpAssignVariableOp assignvariableop_dense_71_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1¥
AssignVariableOp_1AssignVariableOp assignvariableop_1_dense_71_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0	*
_output_shapes
:2

Identity_2¡
AssignVariableOp_2AssignVariableOpassignvariableop_2_adam_iterIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	2
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3£
AssignVariableOp_3AssignVariableOpassignvariableop_3_adam_beta_1Identity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_3k

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:2

Identity_4£
AssignVariableOp_4AssignVariableOpassignvariableop_4_adam_beta_2Identity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_4k

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:2

Identity_5¢
AssignVariableOp_5AssignVariableOpassignvariableop_5_adam_decayIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_5k

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:2

Identity_6ª
AssignVariableOp_6AssignVariableOp%assignvariableop_6_adam_learning_rateIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_6k

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:2

Identity_7³
AssignVariableOp_7AssignVariableOp.assignvariableop_7_lstm_71_lstm_cell_71_kernelIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_7k

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:2

Identity_8½
AssignVariableOp_8AssignVariableOp8assignvariableop_8_lstm_71_lstm_cell_71_recurrent_kernelIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_8k

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:2

Identity_9±
AssignVariableOp_9AssignVariableOp,assignvariableop_9_lstm_71_lstm_cell_71_biasIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_9n
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:2
Identity_10¡
AssignVariableOp_10AssignVariableOpassignvariableop_10_totalIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_10n
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:2
Identity_11¡
AssignVariableOp_11AssignVariableOpassignvariableop_11_countIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_11n
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:2
Identity_12²
AssignVariableOp_12AssignVariableOp*assignvariableop_12_adam_dense_71_kernel_mIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_12n
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:2
Identity_13°
AssignVariableOp_13AssignVariableOp(assignvariableop_13_adam_dense_71_bias_mIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_13n
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:2
Identity_14¾
AssignVariableOp_14AssignVariableOp6assignvariableop_14_adam_lstm_71_lstm_cell_71_kernel_mIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_14n
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:2
Identity_15È
AssignVariableOp_15AssignVariableOp@assignvariableop_15_adam_lstm_71_lstm_cell_71_recurrent_kernel_mIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_15n
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:2
Identity_16¼
AssignVariableOp_16AssignVariableOp4assignvariableop_16_adam_lstm_71_lstm_cell_71_bias_mIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_16n
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:2
Identity_17²
AssignVariableOp_17AssignVariableOp*assignvariableop_17_adam_dense_71_kernel_vIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_17n
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:2
Identity_18°
AssignVariableOp_18AssignVariableOp(assignvariableop_18_adam_dense_71_bias_vIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_18n
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:2
Identity_19¾
AssignVariableOp_19AssignVariableOp6assignvariableop_19_adam_lstm_71_lstm_cell_71_kernel_vIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_19n
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:2
Identity_20È
AssignVariableOp_20AssignVariableOp@assignvariableop_20_adam_lstm_71_lstm_cell_71_recurrent_kernel_vIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_20n
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:2
Identity_21¼
AssignVariableOp_21AssignVariableOp4assignvariableop_21_adam_lstm_71_lstm_cell_71_bias_vIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_219
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOpÂ
Identity_22Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2
Identity_22µ
Identity_23IdentityIdentity_22:output:0^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*
T0*
_output_shapes
: 2
Identity_23"#
identity_23Identity_23:output:0*m
_input_shapes\
Z: ::::::::::::::::::::::2$
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
AssignVariableOp_21AssignVariableOp_212(
AssignVariableOp_3AssignVariableOp_32(
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
C
þ
while_body_624791
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_07
3while_lstm_cell_71_matmul_readvariableop_resource_09
5while_lstm_cell_71_matmul_1_readvariableop_resource_08
4while_lstm_cell_71_biasadd_readvariableop_resource_0
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor5
1while_lstm_cell_71_matmul_readvariableop_resource7
3while_lstm_cell_71_matmul_1_readvariableop_resource6
2while_lstm_cell_71_biasadd_readvariableop_resource¢)while/lstm_cell_71/BiasAdd/ReadVariableOp¢(while/lstm_cell_71/MatMul/ReadVariableOp¢*while/lstm_cell_71/MatMul_1/ReadVariableOpÃ
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   29
7while/TensorArrayV2Read/TensorListGetItem/element_shapeÓ
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItemÉ
(while/lstm_cell_71/MatMul/ReadVariableOpReadVariableOp3while_lstm_cell_71_matmul_readvariableop_resource_0*
_output_shapes
:	*
dtype02*
(while/lstm_cell_71/MatMul/ReadVariableOp×
while/lstm_cell_71/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:00while/lstm_cell_71/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/lstm_cell_71/MatMulÏ
*while/lstm_cell_71/MatMul_1/ReadVariableOpReadVariableOp5while_lstm_cell_71_matmul_1_readvariableop_resource_0*
_output_shapes
:	d*
dtype02,
*while/lstm_cell_71/MatMul_1/ReadVariableOpÀ
while/lstm_cell_71/MatMul_1MatMulwhile_placeholder_22while/lstm_cell_71/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/lstm_cell_71/MatMul_1¸
while/lstm_cell_71/addAddV2#while/lstm_cell_71/MatMul:product:0%while/lstm_cell_71/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/lstm_cell_71/addÈ
)while/lstm_cell_71/BiasAdd/ReadVariableOpReadVariableOp4while_lstm_cell_71_biasadd_readvariableop_resource_0*
_output_shapes	
:*
dtype02+
)while/lstm_cell_71/BiasAdd/ReadVariableOpÅ
while/lstm_cell_71/BiasAddBiasAddwhile/lstm_cell_71/add:z:01while/lstm_cell_71/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/lstm_cell_71/BiasAddv
while/lstm_cell_71/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
while/lstm_cell_71/Const
"while/lstm_cell_71/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2$
"while/lstm_cell_71/split/split_dim
while/lstm_cell_71/splitSplit+while/lstm_cell_71/split/split_dim:output:0#while/lstm_cell_71/BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd*
	num_split2
while/lstm_cell_71/split
while/lstm_cell_71/SigmoidSigmoid!while/lstm_cell_71/split:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd2
while/lstm_cell_71/Sigmoid
while/lstm_cell_71/Sigmoid_1Sigmoid!while/lstm_cell_71/split:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd2
while/lstm_cell_71/Sigmoid_1 
while/lstm_cell_71/mulMul while/lstm_cell_71/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd2
while/lstm_cell_71/mul
while/lstm_cell_71/ReluRelu!while/lstm_cell_71/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd2
while/lstm_cell_71/Relu´
while/lstm_cell_71/mul_1Mulwhile/lstm_cell_71/Sigmoid:y:0%while/lstm_cell_71/Relu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd2
while/lstm_cell_71/mul_1©
while/lstm_cell_71/add_1AddV2while/lstm_cell_71/mul:z:0while/lstm_cell_71/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd2
while/lstm_cell_71/add_1
while/lstm_cell_71/Sigmoid_2Sigmoid!while/lstm_cell_71/split:output:3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd2
while/lstm_cell_71/Sigmoid_2
while/lstm_cell_71/Relu_1Reluwhile/lstm_cell_71/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd2
while/lstm_cell_71/Relu_1¸
while/lstm_cell_71/mul_2Mul while/lstm_cell_71/Sigmoid_2:y:0'while/lstm_cell_71/Relu_1:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd2
while/lstm_cell_71/mul_2à
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_71/mul_2:z:0*
_output_shapes
: *
element_dtype02,
*while/TensorArrayV2Write/TensorListSetItem\
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add/yi
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: 2
	while/add`
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add_1/yv
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: 2
while/add_1â
while/IdentityIdentitywhile/add_1:z:0*^while/lstm_cell_71/BiasAdd/ReadVariableOp)^while/lstm_cell_71/MatMul/ReadVariableOp+^while/lstm_cell_71/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identityõ
while/Identity_1Identitywhile_while_maximum_iterations*^while/lstm_cell_71/BiasAdd/ReadVariableOp)^while/lstm_cell_71/MatMul/ReadVariableOp+^while/lstm_cell_71/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_1ä
while/Identity_2Identitywhile/add:z:0*^while/lstm_cell_71/BiasAdd/ReadVariableOp)^while/lstm_cell_71/MatMul/ReadVariableOp+^while/lstm_cell_71/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_2
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0*^while/lstm_cell_71/BiasAdd/ReadVariableOp)^while/lstm_cell_71/MatMul/ReadVariableOp+^while/lstm_cell_71/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_3
while/Identity_4Identitywhile/lstm_cell_71/mul_2:z:0*^while/lstm_cell_71/BiasAdd/ReadVariableOp)^while/lstm_cell_71/MatMul/ReadVariableOp+^while/lstm_cell_71/MatMul_1/ReadVariableOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd2
while/Identity_4
while/Identity_5Identitywhile/lstm_cell_71/add_1:z:0*^while/lstm_cell_71/BiasAdd/ReadVariableOp)^while/lstm_cell_71/MatMul/ReadVariableOp+^while/lstm_cell_71/MatMul_1/ReadVariableOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd2
while/Identity_5")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"j
2while_lstm_cell_71_biasadd_readvariableop_resource4while_lstm_cell_71_biasadd_readvariableop_resource_0"l
3while_lstm_cell_71_matmul_1_readvariableop_resource5while_lstm_cell_71_matmul_1_readvariableop_resource_0"h
1while_lstm_cell_71_matmul_readvariableop_resource3while_lstm_cell_71_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*Q
_input_shapes@
>: : : : :ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd: : :::2V
)while/lstm_cell_71/BiasAdd/ReadVariableOp)while/lstm_cell_71/BiasAdd/ReadVariableOp2T
(while/lstm_cell_71/MatMul/ReadVariableOp(while/lstm_cell_71/MatMul/ReadVariableOp2X
*while/lstm_cell_71/MatMul_1/ReadVariableOp*while/lstm_cell_71/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd:-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd:

_output_shapes
: :

_output_shapes
: 
ö

I__inference_sequential_71_layer_call_and_return_conditional_losses_624337

inputs
lstm_71_624324
lstm_71_624326
lstm_71_624328
dense_71_624331
dense_71_624333
identity¢ dense_71/StatefulPartitionedCall¢lstm_71/StatefulPartitionedCall¡
lstm_71/StatefulPartitionedCallStatefulPartitionedCallinputslstm_71_624324lstm_71_624326lstm_71_624328*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_lstm_71_layer_call_and_return_conditional_losses_6242142!
lstm_71/StatefulPartitionedCall¶
 dense_71/StatefulPartitionedCallStatefulPartitionedCall(lstm_71/StatefulPartitionedCall:output:0dense_71_624331dense_71_624333*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_dense_71_layer_call_and_return_conditional_losses_6242542"
 dense_71/StatefulPartitionedCallÂ
IdentityIdentity)dense_71/StatefulPartitionedCall:output:0!^dense_71/StatefulPartitionedCall ^lstm_71/StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:ÿÿÿÿÿÿÿÿÿ:::::2D
 dense_71/StatefulPartitionedCall dense_71/StatefulPartitionedCall2B
lstm_71/StatefulPartitionedCalllstm_71/StatefulPartitionedCall:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
«
Ã
while_cond_625271
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_14
0while_while_cond_625271___redundant_placeholder04
0while_while_cond_625271___redundant_placeholder14
0while_while_cond_625271___redundant_placeholder24
0while_while_cond_625271___redundant_placeholder3
while_identity
p

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: 2

while/Less]
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: 2
while/Identity")
while_identitywhile/Identity:output:0*S
_input_shapesB
@: : : : :ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd:-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd:

_output_shapes
: :

_output_shapes
:
«
Ã
while_cond_624790
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_14
0while_while_cond_624790___redundant_placeholder04
0while_while_cond_624790___redundant_placeholder14
0while_while_cond_624790___redundant_placeholder24
0while_while_cond_624790___redundant_placeholder3
while_identity
p

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: 2

while/Less]
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: 2
while/Identity")
while_identitywhile/Identity:output:0*S
_input_shapesB
@: : : : :ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd:-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd:

_output_shapes
: :

_output_shapes
:
«
Û
H__inference_lstm_cell_71_layer_call_and_return_conditional_losses_623368

inputs

states
states_1"
matmul_readvariableop_resource$
 matmul_1_readvariableop_resource#
biasadd_readvariableop_resource
identity

identity_1

identity_2¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp¢MatMul_1/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
MatMul
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource*
_output_shapes
:	d*
dtype02
MatMul_1/ReadVariableOpz
MatMul_1MatMulstatesMatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

MatMul_1l
addAddV2MatMul:product:0MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
add
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype02
BiasAdd/ReadVariableOpy
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
BiasAddP
ConstConst*
_output_shapes
: *
dtype0*
value	B :2
Constd
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split/split_dim¿
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd*
	num_split2
split_
SigmoidSigmoidsplit:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd2	
Sigmoidc
	Sigmoid_1Sigmoidsplit:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd2
	Sigmoid_1\
mulMulSigmoid_1:y:0states_1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd2
mulV
ReluRelusplit:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd2
Reluh
mul_1MulSigmoid:y:0Relu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd2
mul_1]
add_1AddV2mul:z:0	mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd2
add_1c
	Sigmoid_2Sigmoidsplit:output:3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd2
	Sigmoid_2U
Relu_1Relu	add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd2
Relu_1l
mul_2MulSigmoid_2:y:0Relu_1:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd2
mul_2¨
IdentityIdentity	mul_2:z:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd2

Identity¬

Identity_1Identity	mul_2:z:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd2

Identity_1¬

Identity_2Identity	add_1:z:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd2

Identity_2"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*X
_input_shapesG
E:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd:::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs:OK
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
 
_user_specified_namestates:OK
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
 
_user_specified_namestates
«
Û
H__inference_lstm_cell_71_layer_call_and_return_conditional_losses_623401

inputs

states
states_1"
matmul_readvariableop_resource$
 matmul_1_readvariableop_resource#
biasadd_readvariableop_resource
identity

identity_1

identity_2¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp¢MatMul_1/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
MatMul
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource*
_output_shapes
:	d*
dtype02
MatMul_1/ReadVariableOpz
MatMul_1MatMulstatesMatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

MatMul_1l
addAddV2MatMul:product:0MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
add
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype02
BiasAdd/ReadVariableOpy
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
BiasAddP
ConstConst*
_output_shapes
: *
dtype0*
value	B :2
Constd
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split/split_dim¿
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd*
	num_split2
split_
SigmoidSigmoidsplit:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd2	
Sigmoidc
	Sigmoid_1Sigmoidsplit:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd2
	Sigmoid_1\
mulMulSigmoid_1:y:0states_1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd2
mulV
ReluRelusplit:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd2
Reluh
mul_1MulSigmoid:y:0Relu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd2
mul_1]
add_1AddV2mul:z:0	mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd2
add_1c
	Sigmoid_2Sigmoidsplit:output:3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd2
	Sigmoid_2U
Relu_1Relu	add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd2
Relu_1l
mul_2MulSigmoid_2:y:0Relu_1:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd2
mul_2¨
IdentityIdentity	mul_2:z:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd2

Identity¬

Identity_1Identity	mul_2:z:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd2

Identity_1¬

Identity_2Identity	add_1:z:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd2

Identity_2"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*X
_input_shapesG
E:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd:::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs:OK
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
 
_user_specified_namestates:OK
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
 
_user_specified_namestates


I__inference_sequential_71_layer_call_and_return_conditional_losses_624287
lstm_71_input
lstm_71_624274
lstm_71_624276
lstm_71_624278
dense_71_624281
dense_71_624283
identity¢ dense_71/StatefulPartitionedCall¢lstm_71/StatefulPartitionedCall¨
lstm_71/StatefulPartitionedCallStatefulPartitionedCalllstm_71_inputlstm_71_624274lstm_71_624276lstm_71_624278*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_lstm_71_layer_call_and_return_conditional_losses_6242142!
lstm_71/StatefulPartitionedCall¶
 dense_71/StatefulPartitionedCallStatefulPartitionedCall(lstm_71/StatefulPartitionedCall:output:0dense_71_624281dense_71_624283*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_dense_71_layer_call_and_return_conditional_losses_6242542"
 dense_71/StatefulPartitionedCallÂ
IdentityIdentity)dense_71/StatefulPartitionedCall:output:0!^dense_71/StatefulPartitionedCall ^lstm_71/StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:ÿÿÿÿÿÿÿÿÿ:::::2D
 dense_71/StatefulPartitionedCall dense_71/StatefulPartitionedCall2B
lstm_71/StatefulPartitionedCalllstm_71/StatefulPartitionedCall:Z V
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
'
_user_specified_namelstm_71_input


ã
lstm_71_while_cond_624601,
(lstm_71_while_lstm_71_while_loop_counter2
.lstm_71_while_lstm_71_while_maximum_iterations
lstm_71_while_placeholder
lstm_71_while_placeholder_1
lstm_71_while_placeholder_2
lstm_71_while_placeholder_3.
*lstm_71_while_less_lstm_71_strided_slice_1D
@lstm_71_while_lstm_71_while_cond_624601___redundant_placeholder0D
@lstm_71_while_lstm_71_while_cond_624601___redundant_placeholder1D
@lstm_71_while_lstm_71_while_cond_624601___redundant_placeholder2D
@lstm_71_while_lstm_71_while_cond_624601___redundant_placeholder3
lstm_71_while_identity

lstm_71/while/LessLesslstm_71_while_placeholder*lstm_71_while_less_lstm_71_strided_slice_1*
T0*
_output_shapes
: 2
lstm_71/while/Lessu
lstm_71/while/IdentityIdentitylstm_71/while/Less:z:0*
T0
*
_output_shapes
: 2
lstm_71/while/Identity"9
lstm_71_while_identitylstm_71/while/Identity:output:0*S
_input_shapesB
@: : : : :ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd:-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd:

_output_shapes
: :

_output_shapes
:
C
þ
while_body_624944
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_07
3while_lstm_cell_71_matmul_readvariableop_resource_09
5while_lstm_cell_71_matmul_1_readvariableop_resource_08
4while_lstm_cell_71_biasadd_readvariableop_resource_0
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor5
1while_lstm_cell_71_matmul_readvariableop_resource7
3while_lstm_cell_71_matmul_1_readvariableop_resource6
2while_lstm_cell_71_biasadd_readvariableop_resource¢)while/lstm_cell_71/BiasAdd/ReadVariableOp¢(while/lstm_cell_71/MatMul/ReadVariableOp¢*while/lstm_cell_71/MatMul_1/ReadVariableOpÃ
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   29
7while/TensorArrayV2Read/TensorListGetItem/element_shapeÓ
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItemÉ
(while/lstm_cell_71/MatMul/ReadVariableOpReadVariableOp3while_lstm_cell_71_matmul_readvariableop_resource_0*
_output_shapes
:	*
dtype02*
(while/lstm_cell_71/MatMul/ReadVariableOp×
while/lstm_cell_71/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:00while/lstm_cell_71/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/lstm_cell_71/MatMulÏ
*while/lstm_cell_71/MatMul_1/ReadVariableOpReadVariableOp5while_lstm_cell_71_matmul_1_readvariableop_resource_0*
_output_shapes
:	d*
dtype02,
*while/lstm_cell_71/MatMul_1/ReadVariableOpÀ
while/lstm_cell_71/MatMul_1MatMulwhile_placeholder_22while/lstm_cell_71/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/lstm_cell_71/MatMul_1¸
while/lstm_cell_71/addAddV2#while/lstm_cell_71/MatMul:product:0%while/lstm_cell_71/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/lstm_cell_71/addÈ
)while/lstm_cell_71/BiasAdd/ReadVariableOpReadVariableOp4while_lstm_cell_71_biasadd_readvariableop_resource_0*
_output_shapes	
:*
dtype02+
)while/lstm_cell_71/BiasAdd/ReadVariableOpÅ
while/lstm_cell_71/BiasAddBiasAddwhile/lstm_cell_71/add:z:01while/lstm_cell_71/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/lstm_cell_71/BiasAddv
while/lstm_cell_71/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
while/lstm_cell_71/Const
"while/lstm_cell_71/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2$
"while/lstm_cell_71/split/split_dim
while/lstm_cell_71/splitSplit+while/lstm_cell_71/split/split_dim:output:0#while/lstm_cell_71/BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd*
	num_split2
while/lstm_cell_71/split
while/lstm_cell_71/SigmoidSigmoid!while/lstm_cell_71/split:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd2
while/lstm_cell_71/Sigmoid
while/lstm_cell_71/Sigmoid_1Sigmoid!while/lstm_cell_71/split:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd2
while/lstm_cell_71/Sigmoid_1 
while/lstm_cell_71/mulMul while/lstm_cell_71/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd2
while/lstm_cell_71/mul
while/lstm_cell_71/ReluRelu!while/lstm_cell_71/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd2
while/lstm_cell_71/Relu´
while/lstm_cell_71/mul_1Mulwhile/lstm_cell_71/Sigmoid:y:0%while/lstm_cell_71/Relu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd2
while/lstm_cell_71/mul_1©
while/lstm_cell_71/add_1AddV2while/lstm_cell_71/mul:z:0while/lstm_cell_71/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd2
while/lstm_cell_71/add_1
while/lstm_cell_71/Sigmoid_2Sigmoid!while/lstm_cell_71/split:output:3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd2
while/lstm_cell_71/Sigmoid_2
while/lstm_cell_71/Relu_1Reluwhile/lstm_cell_71/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd2
while/lstm_cell_71/Relu_1¸
while/lstm_cell_71/mul_2Mul while/lstm_cell_71/Sigmoid_2:y:0'while/lstm_cell_71/Relu_1:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd2
while/lstm_cell_71/mul_2à
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_71/mul_2:z:0*
_output_shapes
: *
element_dtype02,
*while/TensorArrayV2Write/TensorListSetItem\
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add/yi
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: 2
	while/add`
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add_1/yv
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: 2
while/add_1â
while/IdentityIdentitywhile/add_1:z:0*^while/lstm_cell_71/BiasAdd/ReadVariableOp)^while/lstm_cell_71/MatMul/ReadVariableOp+^while/lstm_cell_71/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identityõ
while/Identity_1Identitywhile_while_maximum_iterations*^while/lstm_cell_71/BiasAdd/ReadVariableOp)^while/lstm_cell_71/MatMul/ReadVariableOp+^while/lstm_cell_71/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_1ä
while/Identity_2Identitywhile/add:z:0*^while/lstm_cell_71/BiasAdd/ReadVariableOp)^while/lstm_cell_71/MatMul/ReadVariableOp+^while/lstm_cell_71/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_2
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0*^while/lstm_cell_71/BiasAdd/ReadVariableOp)^while/lstm_cell_71/MatMul/ReadVariableOp+^while/lstm_cell_71/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_3
while/Identity_4Identitywhile/lstm_cell_71/mul_2:z:0*^while/lstm_cell_71/BiasAdd/ReadVariableOp)^while/lstm_cell_71/MatMul/ReadVariableOp+^while/lstm_cell_71/MatMul_1/ReadVariableOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd2
while/Identity_4
while/Identity_5Identitywhile/lstm_cell_71/add_1:z:0*^while/lstm_cell_71/BiasAdd/ReadVariableOp)^while/lstm_cell_71/MatMul/ReadVariableOp+^while/lstm_cell_71/MatMul_1/ReadVariableOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd2
while/Identity_5")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"j
2while_lstm_cell_71_biasadd_readvariableop_resource4while_lstm_cell_71_biasadd_readvariableop_resource_0"l
3while_lstm_cell_71_matmul_1_readvariableop_resource5while_lstm_cell_71_matmul_1_readvariableop_resource_0"h
1while_lstm_cell_71_matmul_readvariableop_resource3while_lstm_cell_71_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*Q
_input_shapes@
>: : : : :ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd: : :::2V
)while/lstm_cell_71/BiasAdd/ReadVariableOp)while/lstm_cell_71/BiasAdd/ReadVariableOp2T
(while/lstm_cell_71/MatMul/ReadVariableOp(while/lstm_cell_71/MatMul/ReadVariableOp2X
*while/lstm_cell_71/MatMul_1/ReadVariableOp*while/lstm_cell_71/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd:-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd:

_output_shapes
: :

_output_shapes
: 
O
þ	
lstm_71_while_body_624602,
(lstm_71_while_lstm_71_while_loop_counter2
.lstm_71_while_lstm_71_while_maximum_iterations
lstm_71_while_placeholder
lstm_71_while_placeholder_1
lstm_71_while_placeholder_2
lstm_71_while_placeholder_3+
'lstm_71_while_lstm_71_strided_slice_1_0g
clstm_71_while_tensorarrayv2read_tensorlistgetitem_lstm_71_tensorarrayunstack_tensorlistfromtensor_0?
;lstm_71_while_lstm_cell_71_matmul_readvariableop_resource_0A
=lstm_71_while_lstm_cell_71_matmul_1_readvariableop_resource_0@
<lstm_71_while_lstm_cell_71_biasadd_readvariableop_resource_0
lstm_71_while_identity
lstm_71_while_identity_1
lstm_71_while_identity_2
lstm_71_while_identity_3
lstm_71_while_identity_4
lstm_71_while_identity_5)
%lstm_71_while_lstm_71_strided_slice_1e
alstm_71_while_tensorarrayv2read_tensorlistgetitem_lstm_71_tensorarrayunstack_tensorlistfromtensor=
9lstm_71_while_lstm_cell_71_matmul_readvariableop_resource?
;lstm_71_while_lstm_cell_71_matmul_1_readvariableop_resource>
:lstm_71_while_lstm_cell_71_biasadd_readvariableop_resource¢1lstm_71/while/lstm_cell_71/BiasAdd/ReadVariableOp¢0lstm_71/while/lstm_cell_71/MatMul/ReadVariableOp¢2lstm_71/while/lstm_cell_71/MatMul_1/ReadVariableOpÓ
?lstm_71/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   2A
?lstm_71/while/TensorArrayV2Read/TensorListGetItem/element_shape
1lstm_71/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemclstm_71_while_tensorarrayv2read_tensorlistgetitem_lstm_71_tensorarrayunstack_tensorlistfromtensor_0lstm_71_while_placeholderHlstm_71/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
element_dtype023
1lstm_71/while/TensorArrayV2Read/TensorListGetItemá
0lstm_71/while/lstm_cell_71/MatMul/ReadVariableOpReadVariableOp;lstm_71_while_lstm_cell_71_matmul_readvariableop_resource_0*
_output_shapes
:	*
dtype022
0lstm_71/while/lstm_cell_71/MatMul/ReadVariableOp÷
!lstm_71/while/lstm_cell_71/MatMulMatMul8lstm_71/while/TensorArrayV2Read/TensorListGetItem:item:08lstm_71/while/lstm_cell_71/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2#
!lstm_71/while/lstm_cell_71/MatMulç
2lstm_71/while/lstm_cell_71/MatMul_1/ReadVariableOpReadVariableOp=lstm_71_while_lstm_cell_71_matmul_1_readvariableop_resource_0*
_output_shapes
:	d*
dtype024
2lstm_71/while/lstm_cell_71/MatMul_1/ReadVariableOpà
#lstm_71/while/lstm_cell_71/MatMul_1MatMullstm_71_while_placeholder_2:lstm_71/while/lstm_cell_71/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2%
#lstm_71/while/lstm_cell_71/MatMul_1Ø
lstm_71/while/lstm_cell_71/addAddV2+lstm_71/while/lstm_cell_71/MatMul:product:0-lstm_71/while/lstm_cell_71/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2 
lstm_71/while/lstm_cell_71/addà
1lstm_71/while/lstm_cell_71/BiasAdd/ReadVariableOpReadVariableOp<lstm_71_while_lstm_cell_71_biasadd_readvariableop_resource_0*
_output_shapes	
:*
dtype023
1lstm_71/while/lstm_cell_71/BiasAdd/ReadVariableOpå
"lstm_71/while/lstm_cell_71/BiasAddBiasAdd"lstm_71/while/lstm_cell_71/add:z:09lstm_71/while/lstm_cell_71/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2$
"lstm_71/while/lstm_cell_71/BiasAdd
 lstm_71/while/lstm_cell_71/ConstConst*
_output_shapes
: *
dtype0*
value	B :2"
 lstm_71/while/lstm_cell_71/Const
*lstm_71/while/lstm_cell_71/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2,
*lstm_71/while/lstm_cell_71/split/split_dim«
 lstm_71/while/lstm_cell_71/splitSplit3lstm_71/while/lstm_cell_71/split/split_dim:output:0+lstm_71/while/lstm_cell_71/BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd*
	num_split2"
 lstm_71/while/lstm_cell_71/split°
"lstm_71/while/lstm_cell_71/SigmoidSigmoid)lstm_71/while/lstm_cell_71/split:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd2$
"lstm_71/while/lstm_cell_71/Sigmoid´
$lstm_71/while/lstm_cell_71/Sigmoid_1Sigmoid)lstm_71/while/lstm_cell_71/split:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd2&
$lstm_71/while/lstm_cell_71/Sigmoid_1À
lstm_71/while/lstm_cell_71/mulMul(lstm_71/while/lstm_cell_71/Sigmoid_1:y:0lstm_71_while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd2 
lstm_71/while/lstm_cell_71/mul§
lstm_71/while/lstm_cell_71/ReluRelu)lstm_71/while/lstm_cell_71/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd2!
lstm_71/while/lstm_cell_71/ReluÔ
 lstm_71/while/lstm_cell_71/mul_1Mul&lstm_71/while/lstm_cell_71/Sigmoid:y:0-lstm_71/while/lstm_cell_71/Relu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd2"
 lstm_71/while/lstm_cell_71/mul_1É
 lstm_71/while/lstm_cell_71/add_1AddV2"lstm_71/while/lstm_cell_71/mul:z:0$lstm_71/while/lstm_cell_71/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd2"
 lstm_71/while/lstm_cell_71/add_1´
$lstm_71/while/lstm_cell_71/Sigmoid_2Sigmoid)lstm_71/while/lstm_cell_71/split:output:3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd2&
$lstm_71/while/lstm_cell_71/Sigmoid_2¦
!lstm_71/while/lstm_cell_71/Relu_1Relu$lstm_71/while/lstm_cell_71/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd2#
!lstm_71/while/lstm_cell_71/Relu_1Ø
 lstm_71/while/lstm_cell_71/mul_2Mul(lstm_71/while/lstm_cell_71/Sigmoid_2:y:0/lstm_71/while/lstm_cell_71/Relu_1:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd2"
 lstm_71/while/lstm_cell_71/mul_2
2lstm_71/while/TensorArrayV2Write/TensorListSetItemTensorListSetItemlstm_71_while_placeholder_1lstm_71_while_placeholder$lstm_71/while/lstm_cell_71/mul_2:z:0*
_output_shapes
: *
element_dtype024
2lstm_71/while/TensorArrayV2Write/TensorListSetIteml
lstm_71/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_71/while/add/y
lstm_71/while/addAddV2lstm_71_while_placeholderlstm_71/while/add/y:output:0*
T0*
_output_shapes
: 2
lstm_71/while/addp
lstm_71/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_71/while/add_1/y
lstm_71/while/add_1AddV2(lstm_71_while_lstm_71_while_loop_counterlstm_71/while/add_1/y:output:0*
T0*
_output_shapes
: 2
lstm_71/while/add_1
lstm_71/while/IdentityIdentitylstm_71/while/add_1:z:02^lstm_71/while/lstm_cell_71/BiasAdd/ReadVariableOp1^lstm_71/while/lstm_cell_71/MatMul/ReadVariableOp3^lstm_71/while/lstm_cell_71/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
lstm_71/while/Identity­
lstm_71/while/Identity_1Identity.lstm_71_while_lstm_71_while_maximum_iterations2^lstm_71/while/lstm_cell_71/BiasAdd/ReadVariableOp1^lstm_71/while/lstm_cell_71/MatMul/ReadVariableOp3^lstm_71/while/lstm_cell_71/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
lstm_71/while/Identity_1
lstm_71/while/Identity_2Identitylstm_71/while/add:z:02^lstm_71/while/lstm_cell_71/BiasAdd/ReadVariableOp1^lstm_71/while/lstm_cell_71/MatMul/ReadVariableOp3^lstm_71/while/lstm_cell_71/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
lstm_71/while/Identity_2Á
lstm_71/while/Identity_3IdentityBlstm_71/while/TensorArrayV2Write/TensorListSetItem:output_handle:02^lstm_71/while/lstm_cell_71/BiasAdd/ReadVariableOp1^lstm_71/while/lstm_cell_71/MatMul/ReadVariableOp3^lstm_71/while/lstm_cell_71/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
lstm_71/while/Identity_3´
lstm_71/while/Identity_4Identity$lstm_71/while/lstm_cell_71/mul_2:z:02^lstm_71/while/lstm_cell_71/BiasAdd/ReadVariableOp1^lstm_71/while/lstm_cell_71/MatMul/ReadVariableOp3^lstm_71/while/lstm_cell_71/MatMul_1/ReadVariableOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd2
lstm_71/while/Identity_4´
lstm_71/while/Identity_5Identity$lstm_71/while/lstm_cell_71/add_1:z:02^lstm_71/while/lstm_cell_71/BiasAdd/ReadVariableOp1^lstm_71/while/lstm_cell_71/MatMul/ReadVariableOp3^lstm_71/while/lstm_cell_71/MatMul_1/ReadVariableOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd2
lstm_71/while/Identity_5"9
lstm_71_while_identitylstm_71/while/Identity:output:0"=
lstm_71_while_identity_1!lstm_71/while/Identity_1:output:0"=
lstm_71_while_identity_2!lstm_71/while/Identity_2:output:0"=
lstm_71_while_identity_3!lstm_71/while/Identity_3:output:0"=
lstm_71_while_identity_4!lstm_71/while/Identity_4:output:0"=
lstm_71_while_identity_5!lstm_71/while/Identity_5:output:0"P
%lstm_71_while_lstm_71_strided_slice_1'lstm_71_while_lstm_71_strided_slice_1_0"z
:lstm_71_while_lstm_cell_71_biasadd_readvariableop_resource<lstm_71_while_lstm_cell_71_biasadd_readvariableop_resource_0"|
;lstm_71_while_lstm_cell_71_matmul_1_readvariableop_resource=lstm_71_while_lstm_cell_71_matmul_1_readvariableop_resource_0"x
9lstm_71_while_lstm_cell_71_matmul_readvariableop_resource;lstm_71_while_lstm_cell_71_matmul_readvariableop_resource_0"È
alstm_71_while_tensorarrayv2read_tensorlistgetitem_lstm_71_tensorarrayunstack_tensorlistfromtensorclstm_71_while_tensorarrayv2read_tensorlistgetitem_lstm_71_tensorarrayunstack_tensorlistfromtensor_0*Q
_input_shapes@
>: : : : :ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd: : :::2f
1lstm_71/while/lstm_cell_71/BiasAdd/ReadVariableOp1lstm_71/while/lstm_cell_71/BiasAdd/ReadVariableOp2d
0lstm_71/while/lstm_cell_71/MatMul/ReadVariableOp0lstm_71/while/lstm_cell_71/MatMul/ReadVariableOp2h
2lstm_71/while/lstm_cell_71/MatMul_1/ReadVariableOp2lstm_71/while/lstm_cell_71/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd:-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd:

_output_shapes
: :

_output_shapes
: 
Ï
°
.__inference_sequential_71_layer_call_fn_624723

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
identity¢StatefulPartitionedCall 
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3*
Tin

2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*'
_read_only_resource_inputs	
*-
config_proto

CPU

GPU 2J 8 *R
fMRK
I__inference_sequential_71_layer_call_and_return_conditional_losses_6243372
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:ÿÿÿÿÿÿÿÿÿ:::::22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs


(__inference_lstm_71_layer_call_fn_625040
inputs_0
unknown
	unknown_0
	unknown_1
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputs_0unknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_lstm_71_layer_call_and_return_conditional_losses_6237642
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:::22
StatefulPartitionedCallStatefulPartitionedCall:^ Z
4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
inputs/0
«
Ã
while_cond_623694
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_14
0while_while_cond_623694___redundant_placeholder04
0while_while_cond_623694___redundant_placeholder14
0while_while_cond_623694___redundant_placeholder24
0while_while_cond_623694___redundant_placeholder3
while_identity
p

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: 2

while/Less]
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: 2
while/Identity")
while_identitywhile/Identity:output:0*S
_input_shapesB
@: : : : :ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd:-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd:

_output_shapes
: :

_output_shapes
:
ä
·
.__inference_sequential_71_layer_call_fn_624319
lstm_71_input
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
identity¢StatefulPartitionedCall§
StatefulPartitionedCallStatefulPartitionedCalllstm_71_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3*
Tin

2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*'
_read_only_resource_inputs	
*-
config_proto

CPU

GPU 2J 8 *R
fMRK
I__inference_sequential_71_layer_call_and_return_conditional_losses_6243062
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:ÿÿÿÿÿÿÿÿÿ:::::22
StatefulPartitionedCallStatefulPartitionedCall:Z V
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
'
_user_specified_namelstm_71_input
[
ò
C__inference_lstm_71_layer_call_and_return_conditional_losses_625204

inputs/
+lstm_cell_71_matmul_readvariableop_resource1
-lstm_cell_71_matmul_1_readvariableop_resource0
,lstm_cell_71_biasadd_readvariableop_resource
identity¢#lstm_cell_71/BiasAdd/ReadVariableOp¢"lstm_cell_71/MatMul/ReadVariableOp¢$lstm_cell_71/MatMul_1/ReadVariableOp¢whileD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2â
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice\
zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :d2
zeros/mul/yl
	zeros/mulMulstrided_slice:output:0zeros/mul/y:output:0*
T0*
_output_shapes
: 2
	zeros/mul_
zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :è2
zeros/Less/yg

zeros/LessLesszeros/mul:z:0zeros/Less/y:output:0*
T0*
_output_shapes
: 2

zeros/Lessb
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :d2
zeros/packed/1
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros/packed_
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros/Constu
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd2
zeros`
zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B :d2
zeros_1/mul/yr
zeros_1/mulMulstrided_slice:output:0zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
zeros_1/mulc
zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :è2
zeros_1/Less/yo
zeros_1/LessLesszeros_1/mul:z:0zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
zeros_1/Lessf
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :d2
zeros_1/packed/1
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros_1/packedc
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros_1/Const}
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd2	
zeros_1u
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/permz
	transpose	Transposeinputstranspose/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
	transposeO
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:2	
Shape_1x
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2î
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2
TensorArrayV2/element_shape²
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2¿
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   27
5TensorArrayUnstack/TensorListFromTensor/element_shapeø
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02)
'TensorArrayUnstack/TensorListFromTensorx
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2ü
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
shrink_axis_mask2
strided_slice_2µ
"lstm_cell_71/MatMul/ReadVariableOpReadVariableOp+lstm_cell_71_matmul_readvariableop_resource*
_output_shapes
:	*
dtype02$
"lstm_cell_71/MatMul/ReadVariableOp­
lstm_cell_71/MatMulMatMulstrided_slice_2:output:0*lstm_cell_71/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell_71/MatMul»
$lstm_cell_71/MatMul_1/ReadVariableOpReadVariableOp-lstm_cell_71_matmul_1_readvariableop_resource*
_output_shapes
:	d*
dtype02&
$lstm_cell_71/MatMul_1/ReadVariableOp©
lstm_cell_71/MatMul_1MatMulzeros:output:0,lstm_cell_71/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell_71/MatMul_1 
lstm_cell_71/addAddV2lstm_cell_71/MatMul:product:0lstm_cell_71/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell_71/add´
#lstm_cell_71/BiasAdd/ReadVariableOpReadVariableOp,lstm_cell_71_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02%
#lstm_cell_71/BiasAdd/ReadVariableOp­
lstm_cell_71/BiasAddBiasAddlstm_cell_71/add:z:0+lstm_cell_71/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell_71/BiasAddj
lstm_cell_71/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_71/Const~
lstm_cell_71/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_71/split/split_dimó
lstm_cell_71/splitSplit%lstm_cell_71/split/split_dim:output:0lstm_cell_71/BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd*
	num_split2
lstm_cell_71/split
lstm_cell_71/SigmoidSigmoidlstm_cell_71/split:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd2
lstm_cell_71/Sigmoid
lstm_cell_71/Sigmoid_1Sigmoidlstm_cell_71/split:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd2
lstm_cell_71/Sigmoid_1
lstm_cell_71/mulMullstm_cell_71/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd2
lstm_cell_71/mul}
lstm_cell_71/ReluRelulstm_cell_71/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd2
lstm_cell_71/Relu
lstm_cell_71/mul_1Mullstm_cell_71/Sigmoid:y:0lstm_cell_71/Relu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd2
lstm_cell_71/mul_1
lstm_cell_71/add_1AddV2lstm_cell_71/mul:z:0lstm_cell_71/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd2
lstm_cell_71/add_1
lstm_cell_71/Sigmoid_2Sigmoidlstm_cell_71/split:output:3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd2
lstm_cell_71/Sigmoid_2|
lstm_cell_71/Relu_1Relulstm_cell_71/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd2
lstm_cell_71/Relu_1 
lstm_cell_71/mul_2Mullstm_cell_71/Sigmoid_2:y:0!lstm_cell_71/Relu_1:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd2
lstm_cell_71/mul_2
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿd   2
TensorArrayV2_1/element_shape¸
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2_1N
timeConst*
_output_shapes
: *
dtype0*
value	B : 2
time
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counterî
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0+lstm_cell_71_matmul_readvariableop_resource-lstm_cell_71_matmul_1_readvariableop_resource,lstm_cell_71_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd: : : : : *%
_read_only_resource_inputs
	
*
bodyR
while_body_625119*
condR
while_cond_625118*K
output_shapes:
8: : : : :ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd: : : : : *
parallel_iterations 2
whileµ
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿd   22
0TensorArrayV2Stack/TensorListStack/element_shapeè
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿd*
element_dtype02$
"TensorArrayV2Stack/TensorListStack
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿ2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd*
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/perm¥
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿd2
transpose_1f
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2	
runtimeæ
IdentityIdentitystrided_slice_3:output:0$^lstm_cell_71/BiasAdd/ReadVariableOp#^lstm_cell_71/MatMul/ReadVariableOp%^lstm_cell_71/MatMul_1/ReadVariableOp^while*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:ÿÿÿÿÿÿÿÿÿ:::2J
#lstm_cell_71/BiasAdd/ReadVariableOp#lstm_cell_71/BiasAdd/ReadVariableOp2H
"lstm_cell_71/MatMul/ReadVariableOp"lstm_cell_71/MatMul/ReadVariableOp2L
$lstm_cell_71/MatMul_1/ReadVariableOp$lstm_cell_71/MatMul_1/ReadVariableOp2
whilewhile:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
[
ò
C__inference_lstm_71_layer_call_and_return_conditional_losses_624061

inputs/
+lstm_cell_71_matmul_readvariableop_resource1
-lstm_cell_71_matmul_1_readvariableop_resource0
,lstm_cell_71_biasadd_readvariableop_resource
identity¢#lstm_cell_71/BiasAdd/ReadVariableOp¢"lstm_cell_71/MatMul/ReadVariableOp¢$lstm_cell_71/MatMul_1/ReadVariableOp¢whileD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2â
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice\
zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :d2
zeros/mul/yl
	zeros/mulMulstrided_slice:output:0zeros/mul/y:output:0*
T0*
_output_shapes
: 2
	zeros/mul_
zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :è2
zeros/Less/yg

zeros/LessLesszeros/mul:z:0zeros/Less/y:output:0*
T0*
_output_shapes
: 2

zeros/Lessb
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :d2
zeros/packed/1
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros/packed_
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros/Constu
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd2
zeros`
zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B :d2
zeros_1/mul/yr
zeros_1/mulMulstrided_slice:output:0zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
zeros_1/mulc
zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :è2
zeros_1/Less/yo
zeros_1/LessLesszeros_1/mul:z:0zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
zeros_1/Lessf
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :d2
zeros_1/packed/1
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros_1/packedc
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros_1/Const}
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd2	
zeros_1u
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/permz
	transpose	Transposeinputstranspose/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
	transposeO
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:2	
Shape_1x
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2î
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2
TensorArrayV2/element_shape²
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2¿
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   27
5TensorArrayUnstack/TensorListFromTensor/element_shapeø
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02)
'TensorArrayUnstack/TensorListFromTensorx
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2ü
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
shrink_axis_mask2
strided_slice_2µ
"lstm_cell_71/MatMul/ReadVariableOpReadVariableOp+lstm_cell_71_matmul_readvariableop_resource*
_output_shapes
:	*
dtype02$
"lstm_cell_71/MatMul/ReadVariableOp­
lstm_cell_71/MatMulMatMulstrided_slice_2:output:0*lstm_cell_71/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell_71/MatMul»
$lstm_cell_71/MatMul_1/ReadVariableOpReadVariableOp-lstm_cell_71_matmul_1_readvariableop_resource*
_output_shapes
:	d*
dtype02&
$lstm_cell_71/MatMul_1/ReadVariableOp©
lstm_cell_71/MatMul_1MatMulzeros:output:0,lstm_cell_71/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell_71/MatMul_1 
lstm_cell_71/addAddV2lstm_cell_71/MatMul:product:0lstm_cell_71/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell_71/add´
#lstm_cell_71/BiasAdd/ReadVariableOpReadVariableOp,lstm_cell_71_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02%
#lstm_cell_71/BiasAdd/ReadVariableOp­
lstm_cell_71/BiasAddBiasAddlstm_cell_71/add:z:0+lstm_cell_71/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell_71/BiasAddj
lstm_cell_71/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_71/Const~
lstm_cell_71/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_71/split/split_dimó
lstm_cell_71/splitSplit%lstm_cell_71/split/split_dim:output:0lstm_cell_71/BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd*
	num_split2
lstm_cell_71/split
lstm_cell_71/SigmoidSigmoidlstm_cell_71/split:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd2
lstm_cell_71/Sigmoid
lstm_cell_71/Sigmoid_1Sigmoidlstm_cell_71/split:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd2
lstm_cell_71/Sigmoid_1
lstm_cell_71/mulMullstm_cell_71/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd2
lstm_cell_71/mul}
lstm_cell_71/ReluRelulstm_cell_71/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd2
lstm_cell_71/Relu
lstm_cell_71/mul_1Mullstm_cell_71/Sigmoid:y:0lstm_cell_71/Relu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd2
lstm_cell_71/mul_1
lstm_cell_71/add_1AddV2lstm_cell_71/mul:z:0lstm_cell_71/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd2
lstm_cell_71/add_1
lstm_cell_71/Sigmoid_2Sigmoidlstm_cell_71/split:output:3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd2
lstm_cell_71/Sigmoid_2|
lstm_cell_71/Relu_1Relulstm_cell_71/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd2
lstm_cell_71/Relu_1 
lstm_cell_71/mul_2Mullstm_cell_71/Sigmoid_2:y:0!lstm_cell_71/Relu_1:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd2
lstm_cell_71/mul_2
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿd   2
TensorArrayV2_1/element_shape¸
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2_1N
timeConst*
_output_shapes
: *
dtype0*
value	B : 2
time
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counterî
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0+lstm_cell_71_matmul_readvariableop_resource-lstm_cell_71_matmul_1_readvariableop_resource,lstm_cell_71_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd: : : : : *%
_read_only_resource_inputs
	
*
bodyR
while_body_623976*
condR
while_cond_623975*K
output_shapes:
8: : : : :ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd: : : : : *
parallel_iterations 2
whileµ
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿd   22
0TensorArrayV2Stack/TensorListStack/element_shapeè
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿd*
element_dtype02$
"TensorArrayV2Stack/TensorListStack
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿ2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd*
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/perm¥
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿd2
transpose_1f
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2	
runtimeæ
IdentityIdentitystrided_slice_3:output:0$^lstm_cell_71/BiasAdd/ReadVariableOp#^lstm_cell_71/MatMul/ReadVariableOp%^lstm_cell_71/MatMul_1/ReadVariableOp^while*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:ÿÿÿÿÿÿÿÿÿ:::2J
#lstm_cell_71/BiasAdd/ReadVariableOp#lstm_cell_71/BiasAdd/ReadVariableOp2H
"lstm_cell_71/MatMul/ReadVariableOp"lstm_cell_71/MatMul/ReadVariableOp2L
$lstm_cell_71/MatMul_1/ReadVariableOp$lstm_cell_71/MatMul_1/ReadVariableOp2
whilewhile:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
«
Ã
while_cond_625118
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_14
0while_while_cond_625118___redundant_placeholder04
0while_while_cond_625118___redundant_placeholder14
0while_while_cond_625118___redundant_placeholder24
0while_while_cond_625118___redundant_placeholder3
while_identity
p

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: 2

while/Less]
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: 2
while/Identity")
while_identitywhile/Identity:output:0*S
_input_shapesB
@: : : : :ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd:-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd:

_output_shapes
: :

_output_shapes
:
²
­
$__inference_signature_wrapper_624375
lstm_71_input
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
identity¢StatefulPartitionedCallÿ
StatefulPartitionedCallStatefulPartitionedCalllstm_71_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3*
Tin

2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*'
_read_only_resource_inputs	
*-
config_proto

CPU

GPU 2J 8 **
f%R#
!__inference__wrapped_model_6232952
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:ÿÿÿÿÿÿÿÿÿ:::::22
StatefulPartitionedCallStatefulPartitionedCall:Z V
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
'
_user_specified_namelstm_71_input


I__inference_sequential_71_layer_call_and_return_conditional_losses_624271
lstm_71_input
lstm_71_624237
lstm_71_624239
lstm_71_624241
dense_71_624265
dense_71_624267
identity¢ dense_71/StatefulPartitionedCall¢lstm_71/StatefulPartitionedCall¨
lstm_71/StatefulPartitionedCallStatefulPartitionedCalllstm_71_inputlstm_71_624237lstm_71_624239lstm_71_624241*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_lstm_71_layer_call_and_return_conditional_losses_6240612!
lstm_71/StatefulPartitionedCall¶
 dense_71/StatefulPartitionedCallStatefulPartitionedCall(lstm_71/StatefulPartitionedCall:output:0dense_71_624265dense_71_624267*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_dense_71_layer_call_and_return_conditional_losses_6242542"
 dense_71/StatefulPartitionedCallÂ
IdentityIdentity)dense_71/StatefulPartitionedCall:output:0!^dense_71/StatefulPartitionedCall ^lstm_71/StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:ÿÿÿÿÿÿÿÿÿ:::::2D
 dense_71/StatefulPartitionedCall dense_71/StatefulPartitionedCall2B
lstm_71/StatefulPartitionedCalllstm_71/StatefulPartitionedCall:Z V
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
'
_user_specified_namelstm_71_input
ö

I__inference_sequential_71_layer_call_and_return_conditional_losses_624306

inputs
lstm_71_624293
lstm_71_624295
lstm_71_624297
dense_71_624300
dense_71_624302
identity¢ dense_71/StatefulPartitionedCall¢lstm_71/StatefulPartitionedCall¡
lstm_71/StatefulPartitionedCallStatefulPartitionedCallinputslstm_71_624293lstm_71_624295lstm_71_624297*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_lstm_71_layer_call_and_return_conditional_losses_6240612!
lstm_71/StatefulPartitionedCall¶
 dense_71/StatefulPartitionedCallStatefulPartitionedCall(lstm_71/StatefulPartitionedCall:output:0dense_71_624300dense_71_624302*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_dense_71_layer_call_and_return_conditional_losses_6242542"
 dense_71/StatefulPartitionedCallÂ
IdentityIdentity)dense_71/StatefulPartitionedCall:output:0!^dense_71/StatefulPartitionedCall ^lstm_71/StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:ÿÿÿÿÿÿÿÿÿ:::::2D
 dense_71/StatefulPartitionedCall dense_71/StatefulPartitionedCall2B
lstm_71/StatefulPartitionedCalllstm_71/StatefulPartitionedCall:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ÛD
Ü
C__inference_lstm_71_layer_call_and_return_conditional_losses_623764

inputs
lstm_cell_71_623682
lstm_cell_71_623684
lstm_cell_71_623686
identity¢$lstm_cell_71/StatefulPartitionedCall¢whileD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2â
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice\
zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :d2
zeros/mul/yl
	zeros/mulMulstrided_slice:output:0zeros/mul/y:output:0*
T0*
_output_shapes
: 2
	zeros/mul_
zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :è2
zeros/Less/yg

zeros/LessLesszeros/mul:z:0zeros/Less/y:output:0*
T0*
_output_shapes
: 2

zeros/Lessb
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :d2
zeros/packed/1
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros/packed_
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros/Constu
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd2
zeros`
zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B :d2
zeros_1/mul/yr
zeros_1/mulMulstrided_slice:output:0zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
zeros_1/mulc
zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :è2
zeros_1/Less/yo
zeros_1/LessLesszeros_1/mul:z:0zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
zeros_1/Lessf
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :d2
zeros_1/packed/1
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros_1/packedc
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros_1/Const}
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd2	
zeros_1u
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/perm
	transpose	Transposeinputstranspose/perm:output:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
	transposeO
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:2	
Shape_1x
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2î
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2
TensorArrayV2/element_shape²
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2¿
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   27
5TensorArrayUnstack/TensorListFromTensor/element_shapeø
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02)
'TensorArrayUnstack/TensorListFromTensorx
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2ü
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
shrink_axis_mask2
strided_slice_2
$lstm_cell_71/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_2:output:0zeros:output:0zeros_1:output:0lstm_cell_71_623682lstm_cell_71_623684lstm_cell_71_623686*
Tin

2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *Q
fLRJ
H__inference_lstm_cell_71_layer_call_and_return_conditional_losses_6233682&
$lstm_cell_71/StatefulPartitionedCall
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿd   2
TensorArrayV2_1/element_shape¸
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2_1N
timeConst*
_output_shapes
: *
dtype0*
value	B : 2
time
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counter£
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0lstm_cell_71_623682lstm_cell_71_623684lstm_cell_71_623686*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd: : : : : *%
_read_only_resource_inputs
	
*
bodyR
while_body_623695*
condR
while_cond_623694*K
output_shapes:
8: : : : :ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd: : : : : *
parallel_iterations 2
whileµ
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿd   22
0TensorArrayV2Stack/TensorListStack/element_shapeñ
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿd*
element_dtype02$
"TensorArrayV2Stack/TensorListStack
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿ2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd*
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/perm®
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿd2
transpose_1f
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2	
runtime
IdentityIdentitystrided_slice_3:output:0%^lstm_cell_71/StatefulPartitionedCall^while*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:::2L
$lstm_cell_71/StatefulPartitionedCall$lstm_cell_71/StatefulPartitionedCall2
whilewhile:\ X
4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ü
~
)__inference_dense_71_layer_call_fn_625398

inputs
unknown
	unknown_0
identity¢StatefulPartitionedCallô
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_dense_71_layer_call_and_return_conditional_losses_6242542
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿd::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
 
_user_specified_nameinputs
O
þ	
lstm_71_while_body_624443,
(lstm_71_while_lstm_71_while_loop_counter2
.lstm_71_while_lstm_71_while_maximum_iterations
lstm_71_while_placeholder
lstm_71_while_placeholder_1
lstm_71_while_placeholder_2
lstm_71_while_placeholder_3+
'lstm_71_while_lstm_71_strided_slice_1_0g
clstm_71_while_tensorarrayv2read_tensorlistgetitem_lstm_71_tensorarrayunstack_tensorlistfromtensor_0?
;lstm_71_while_lstm_cell_71_matmul_readvariableop_resource_0A
=lstm_71_while_lstm_cell_71_matmul_1_readvariableop_resource_0@
<lstm_71_while_lstm_cell_71_biasadd_readvariableop_resource_0
lstm_71_while_identity
lstm_71_while_identity_1
lstm_71_while_identity_2
lstm_71_while_identity_3
lstm_71_while_identity_4
lstm_71_while_identity_5)
%lstm_71_while_lstm_71_strided_slice_1e
alstm_71_while_tensorarrayv2read_tensorlistgetitem_lstm_71_tensorarrayunstack_tensorlistfromtensor=
9lstm_71_while_lstm_cell_71_matmul_readvariableop_resource?
;lstm_71_while_lstm_cell_71_matmul_1_readvariableop_resource>
:lstm_71_while_lstm_cell_71_biasadd_readvariableop_resource¢1lstm_71/while/lstm_cell_71/BiasAdd/ReadVariableOp¢0lstm_71/while/lstm_cell_71/MatMul/ReadVariableOp¢2lstm_71/while/lstm_cell_71/MatMul_1/ReadVariableOpÓ
?lstm_71/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   2A
?lstm_71/while/TensorArrayV2Read/TensorListGetItem/element_shape
1lstm_71/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemclstm_71_while_tensorarrayv2read_tensorlistgetitem_lstm_71_tensorarrayunstack_tensorlistfromtensor_0lstm_71_while_placeholderHlstm_71/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
element_dtype023
1lstm_71/while/TensorArrayV2Read/TensorListGetItemá
0lstm_71/while/lstm_cell_71/MatMul/ReadVariableOpReadVariableOp;lstm_71_while_lstm_cell_71_matmul_readvariableop_resource_0*
_output_shapes
:	*
dtype022
0lstm_71/while/lstm_cell_71/MatMul/ReadVariableOp÷
!lstm_71/while/lstm_cell_71/MatMulMatMul8lstm_71/while/TensorArrayV2Read/TensorListGetItem:item:08lstm_71/while/lstm_cell_71/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2#
!lstm_71/while/lstm_cell_71/MatMulç
2lstm_71/while/lstm_cell_71/MatMul_1/ReadVariableOpReadVariableOp=lstm_71_while_lstm_cell_71_matmul_1_readvariableop_resource_0*
_output_shapes
:	d*
dtype024
2lstm_71/while/lstm_cell_71/MatMul_1/ReadVariableOpà
#lstm_71/while/lstm_cell_71/MatMul_1MatMullstm_71_while_placeholder_2:lstm_71/while/lstm_cell_71/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2%
#lstm_71/while/lstm_cell_71/MatMul_1Ø
lstm_71/while/lstm_cell_71/addAddV2+lstm_71/while/lstm_cell_71/MatMul:product:0-lstm_71/while/lstm_cell_71/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2 
lstm_71/while/lstm_cell_71/addà
1lstm_71/while/lstm_cell_71/BiasAdd/ReadVariableOpReadVariableOp<lstm_71_while_lstm_cell_71_biasadd_readvariableop_resource_0*
_output_shapes	
:*
dtype023
1lstm_71/while/lstm_cell_71/BiasAdd/ReadVariableOpå
"lstm_71/while/lstm_cell_71/BiasAddBiasAdd"lstm_71/while/lstm_cell_71/add:z:09lstm_71/while/lstm_cell_71/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2$
"lstm_71/while/lstm_cell_71/BiasAdd
 lstm_71/while/lstm_cell_71/ConstConst*
_output_shapes
: *
dtype0*
value	B :2"
 lstm_71/while/lstm_cell_71/Const
*lstm_71/while/lstm_cell_71/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2,
*lstm_71/while/lstm_cell_71/split/split_dim«
 lstm_71/while/lstm_cell_71/splitSplit3lstm_71/while/lstm_cell_71/split/split_dim:output:0+lstm_71/while/lstm_cell_71/BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd*
	num_split2"
 lstm_71/while/lstm_cell_71/split°
"lstm_71/while/lstm_cell_71/SigmoidSigmoid)lstm_71/while/lstm_cell_71/split:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd2$
"lstm_71/while/lstm_cell_71/Sigmoid´
$lstm_71/while/lstm_cell_71/Sigmoid_1Sigmoid)lstm_71/while/lstm_cell_71/split:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd2&
$lstm_71/while/lstm_cell_71/Sigmoid_1À
lstm_71/while/lstm_cell_71/mulMul(lstm_71/while/lstm_cell_71/Sigmoid_1:y:0lstm_71_while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd2 
lstm_71/while/lstm_cell_71/mul§
lstm_71/while/lstm_cell_71/ReluRelu)lstm_71/while/lstm_cell_71/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd2!
lstm_71/while/lstm_cell_71/ReluÔ
 lstm_71/while/lstm_cell_71/mul_1Mul&lstm_71/while/lstm_cell_71/Sigmoid:y:0-lstm_71/while/lstm_cell_71/Relu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd2"
 lstm_71/while/lstm_cell_71/mul_1É
 lstm_71/while/lstm_cell_71/add_1AddV2"lstm_71/while/lstm_cell_71/mul:z:0$lstm_71/while/lstm_cell_71/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd2"
 lstm_71/while/lstm_cell_71/add_1´
$lstm_71/while/lstm_cell_71/Sigmoid_2Sigmoid)lstm_71/while/lstm_cell_71/split:output:3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd2&
$lstm_71/while/lstm_cell_71/Sigmoid_2¦
!lstm_71/while/lstm_cell_71/Relu_1Relu$lstm_71/while/lstm_cell_71/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd2#
!lstm_71/while/lstm_cell_71/Relu_1Ø
 lstm_71/while/lstm_cell_71/mul_2Mul(lstm_71/while/lstm_cell_71/Sigmoid_2:y:0/lstm_71/while/lstm_cell_71/Relu_1:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd2"
 lstm_71/while/lstm_cell_71/mul_2
2lstm_71/while/TensorArrayV2Write/TensorListSetItemTensorListSetItemlstm_71_while_placeholder_1lstm_71_while_placeholder$lstm_71/while/lstm_cell_71/mul_2:z:0*
_output_shapes
: *
element_dtype024
2lstm_71/while/TensorArrayV2Write/TensorListSetIteml
lstm_71/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_71/while/add/y
lstm_71/while/addAddV2lstm_71_while_placeholderlstm_71/while/add/y:output:0*
T0*
_output_shapes
: 2
lstm_71/while/addp
lstm_71/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_71/while/add_1/y
lstm_71/while/add_1AddV2(lstm_71_while_lstm_71_while_loop_counterlstm_71/while/add_1/y:output:0*
T0*
_output_shapes
: 2
lstm_71/while/add_1
lstm_71/while/IdentityIdentitylstm_71/while/add_1:z:02^lstm_71/while/lstm_cell_71/BiasAdd/ReadVariableOp1^lstm_71/while/lstm_cell_71/MatMul/ReadVariableOp3^lstm_71/while/lstm_cell_71/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
lstm_71/while/Identity­
lstm_71/while/Identity_1Identity.lstm_71_while_lstm_71_while_maximum_iterations2^lstm_71/while/lstm_cell_71/BiasAdd/ReadVariableOp1^lstm_71/while/lstm_cell_71/MatMul/ReadVariableOp3^lstm_71/while/lstm_cell_71/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
lstm_71/while/Identity_1
lstm_71/while/Identity_2Identitylstm_71/while/add:z:02^lstm_71/while/lstm_cell_71/BiasAdd/ReadVariableOp1^lstm_71/while/lstm_cell_71/MatMul/ReadVariableOp3^lstm_71/while/lstm_cell_71/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
lstm_71/while/Identity_2Á
lstm_71/while/Identity_3IdentityBlstm_71/while/TensorArrayV2Write/TensorListSetItem:output_handle:02^lstm_71/while/lstm_cell_71/BiasAdd/ReadVariableOp1^lstm_71/while/lstm_cell_71/MatMul/ReadVariableOp3^lstm_71/while/lstm_cell_71/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
lstm_71/while/Identity_3´
lstm_71/while/Identity_4Identity$lstm_71/while/lstm_cell_71/mul_2:z:02^lstm_71/while/lstm_cell_71/BiasAdd/ReadVariableOp1^lstm_71/while/lstm_cell_71/MatMul/ReadVariableOp3^lstm_71/while/lstm_cell_71/MatMul_1/ReadVariableOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd2
lstm_71/while/Identity_4´
lstm_71/while/Identity_5Identity$lstm_71/while/lstm_cell_71/add_1:z:02^lstm_71/while/lstm_cell_71/BiasAdd/ReadVariableOp1^lstm_71/while/lstm_cell_71/MatMul/ReadVariableOp3^lstm_71/while/lstm_cell_71/MatMul_1/ReadVariableOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd2
lstm_71/while/Identity_5"9
lstm_71_while_identitylstm_71/while/Identity:output:0"=
lstm_71_while_identity_1!lstm_71/while/Identity_1:output:0"=
lstm_71_while_identity_2!lstm_71/while/Identity_2:output:0"=
lstm_71_while_identity_3!lstm_71/while/Identity_3:output:0"=
lstm_71_while_identity_4!lstm_71/while/Identity_4:output:0"=
lstm_71_while_identity_5!lstm_71/while/Identity_5:output:0"P
%lstm_71_while_lstm_71_strided_slice_1'lstm_71_while_lstm_71_strided_slice_1_0"z
:lstm_71_while_lstm_cell_71_biasadd_readvariableop_resource<lstm_71_while_lstm_cell_71_biasadd_readvariableop_resource_0"|
;lstm_71_while_lstm_cell_71_matmul_1_readvariableop_resource=lstm_71_while_lstm_cell_71_matmul_1_readvariableop_resource_0"x
9lstm_71_while_lstm_cell_71_matmul_readvariableop_resource;lstm_71_while_lstm_cell_71_matmul_readvariableop_resource_0"È
alstm_71_while_tensorarrayv2read_tensorlistgetitem_lstm_71_tensorarrayunstack_tensorlistfromtensorclstm_71_while_tensorarrayv2read_tensorlistgetitem_lstm_71_tensorarrayunstack_tensorlistfromtensor_0*Q
_input_shapes@
>: : : : :ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd: : :::2f
1lstm_71/while/lstm_cell_71/BiasAdd/ReadVariableOp1lstm_71/while/lstm_cell_71/BiasAdd/ReadVariableOp2d
0lstm_71/while/lstm_cell_71/MatMul/ReadVariableOp0lstm_71/while/lstm_cell_71/MatMul/ReadVariableOp2h
2lstm_71/while/lstm_cell_71/MatMul_1/ReadVariableOp2lstm_71/while/lstm_cell_71/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd:-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd:

_output_shapes
: :

_output_shapes
: 
«
Ã
while_cond_624943
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_14
0while_while_cond_624943___redundant_placeholder04
0while_while_cond_624943___redundant_placeholder14
0while_while_cond_624943___redundant_placeholder24
0while_while_cond_624943___redundant_placeholder3
while_identity
p

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: 2

while/Less]
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: 2
while/Identity")
while_identitywhile/Identity:output:0*S
_input_shapesB
@: : : : :ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd:-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd:

_output_shapes
: :

_output_shapes
:


ã
lstm_71_while_cond_624442,
(lstm_71_while_lstm_71_while_loop_counter2
.lstm_71_while_lstm_71_while_maximum_iterations
lstm_71_while_placeholder
lstm_71_while_placeholder_1
lstm_71_while_placeholder_2
lstm_71_while_placeholder_3.
*lstm_71_while_less_lstm_71_strided_slice_1D
@lstm_71_while_lstm_71_while_cond_624442___redundant_placeholder0D
@lstm_71_while_lstm_71_while_cond_624442___redundant_placeholder1D
@lstm_71_while_lstm_71_while_cond_624442___redundant_placeholder2D
@lstm_71_while_lstm_71_while_cond_624442___redundant_placeholder3
lstm_71_while_identity

lstm_71/while/LessLesslstm_71_while_placeholder*lstm_71_while_less_lstm_71_strided_slice_1*
T0*
_output_shapes
: 2
lstm_71/while/Lessu
lstm_71/while/IdentityIdentitylstm_71/while/Less:z:0*
T0
*
_output_shapes
: 2
lstm_71/while/Identity"9
lstm_71_while_identitylstm_71/while/Identity:output:0*S
_input_shapesB
@: : : : :ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd:-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd:

_output_shapes
: :

_output_shapes
:
Ç[
ô
C__inference_lstm_71_layer_call_and_return_conditional_losses_624876
inputs_0/
+lstm_cell_71_matmul_readvariableop_resource1
-lstm_cell_71_matmul_1_readvariableop_resource0
,lstm_cell_71_biasadd_readvariableop_resource
identity¢#lstm_cell_71/BiasAdd/ReadVariableOp¢"lstm_cell_71/MatMul/ReadVariableOp¢$lstm_cell_71/MatMul_1/ReadVariableOp¢whileF
ShapeShapeinputs_0*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2â
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice\
zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :d2
zeros/mul/yl
	zeros/mulMulstrided_slice:output:0zeros/mul/y:output:0*
T0*
_output_shapes
: 2
	zeros/mul_
zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :è2
zeros/Less/yg

zeros/LessLesszeros/mul:z:0zeros/Less/y:output:0*
T0*
_output_shapes
: 2

zeros/Lessb
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :d2
zeros/packed/1
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros/packed_
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros/Constu
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd2
zeros`
zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B :d2
zeros_1/mul/yr
zeros_1/mulMulstrided_slice:output:0zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
zeros_1/mulc
zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :è2
zeros_1/Less/yo
zeros_1/LessLesszeros_1/mul:z:0zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
zeros_1/Lessf
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :d2
zeros_1/packed/1
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros_1/packedc
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros_1/Const}
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd2	
zeros_1u
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/perm
	transpose	Transposeinputs_0transpose/perm:output:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
	transposeO
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:2	
Shape_1x
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2î
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2
TensorArrayV2/element_shape²
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2¿
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   27
5TensorArrayUnstack/TensorListFromTensor/element_shapeø
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02)
'TensorArrayUnstack/TensorListFromTensorx
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2ü
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
shrink_axis_mask2
strided_slice_2µ
"lstm_cell_71/MatMul/ReadVariableOpReadVariableOp+lstm_cell_71_matmul_readvariableop_resource*
_output_shapes
:	*
dtype02$
"lstm_cell_71/MatMul/ReadVariableOp­
lstm_cell_71/MatMulMatMulstrided_slice_2:output:0*lstm_cell_71/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell_71/MatMul»
$lstm_cell_71/MatMul_1/ReadVariableOpReadVariableOp-lstm_cell_71_matmul_1_readvariableop_resource*
_output_shapes
:	d*
dtype02&
$lstm_cell_71/MatMul_1/ReadVariableOp©
lstm_cell_71/MatMul_1MatMulzeros:output:0,lstm_cell_71/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell_71/MatMul_1 
lstm_cell_71/addAddV2lstm_cell_71/MatMul:product:0lstm_cell_71/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell_71/add´
#lstm_cell_71/BiasAdd/ReadVariableOpReadVariableOp,lstm_cell_71_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02%
#lstm_cell_71/BiasAdd/ReadVariableOp­
lstm_cell_71/BiasAddBiasAddlstm_cell_71/add:z:0+lstm_cell_71/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell_71/BiasAddj
lstm_cell_71/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_71/Const~
lstm_cell_71/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_71/split/split_dimó
lstm_cell_71/splitSplit%lstm_cell_71/split/split_dim:output:0lstm_cell_71/BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd*
	num_split2
lstm_cell_71/split
lstm_cell_71/SigmoidSigmoidlstm_cell_71/split:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd2
lstm_cell_71/Sigmoid
lstm_cell_71/Sigmoid_1Sigmoidlstm_cell_71/split:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd2
lstm_cell_71/Sigmoid_1
lstm_cell_71/mulMullstm_cell_71/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd2
lstm_cell_71/mul}
lstm_cell_71/ReluRelulstm_cell_71/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd2
lstm_cell_71/Relu
lstm_cell_71/mul_1Mullstm_cell_71/Sigmoid:y:0lstm_cell_71/Relu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd2
lstm_cell_71/mul_1
lstm_cell_71/add_1AddV2lstm_cell_71/mul:z:0lstm_cell_71/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd2
lstm_cell_71/add_1
lstm_cell_71/Sigmoid_2Sigmoidlstm_cell_71/split:output:3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd2
lstm_cell_71/Sigmoid_2|
lstm_cell_71/Relu_1Relulstm_cell_71/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd2
lstm_cell_71/Relu_1 
lstm_cell_71/mul_2Mullstm_cell_71/Sigmoid_2:y:0!lstm_cell_71/Relu_1:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd2
lstm_cell_71/mul_2
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿd   2
TensorArrayV2_1/element_shape¸
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2_1N
timeConst*
_output_shapes
: *
dtype0*
value	B : 2
time
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counterî
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0+lstm_cell_71_matmul_readvariableop_resource-lstm_cell_71_matmul_1_readvariableop_resource,lstm_cell_71_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd: : : : : *%
_read_only_resource_inputs
	
*
bodyR
while_body_624791*
condR
while_cond_624790*K
output_shapes:
8: : : : :ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd: : : : : *
parallel_iterations 2
whileµ
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿd   22
0TensorArrayV2Stack/TensorListStack/element_shapeñ
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿd*
element_dtype02$
"TensorArrayV2Stack/TensorListStack
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿ2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd*
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/perm®
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿd2
transpose_1f
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2	
runtimeæ
IdentityIdentitystrided_slice_3:output:0$^lstm_cell_71/BiasAdd/ReadVariableOp#^lstm_cell_71/MatMul/ReadVariableOp%^lstm_cell_71/MatMul_1/ReadVariableOp^while*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:::2J
#lstm_cell_71/BiasAdd/ReadVariableOp#lstm_cell_71/BiasAdd/ReadVariableOp2H
"lstm_cell_71/MatMul/ReadVariableOp"lstm_cell_71/MatMul/ReadVariableOp2L
$lstm_cell_71/MatMul_1/ReadVariableOp$lstm_cell_71/MatMul_1/ReadVariableOp2
whilewhile:^ Z
4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
inputs/0
[
ò
C__inference_lstm_71_layer_call_and_return_conditional_losses_624214

inputs/
+lstm_cell_71_matmul_readvariableop_resource1
-lstm_cell_71_matmul_1_readvariableop_resource0
,lstm_cell_71_biasadd_readvariableop_resource
identity¢#lstm_cell_71/BiasAdd/ReadVariableOp¢"lstm_cell_71/MatMul/ReadVariableOp¢$lstm_cell_71/MatMul_1/ReadVariableOp¢whileD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2â
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice\
zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :d2
zeros/mul/yl
	zeros/mulMulstrided_slice:output:0zeros/mul/y:output:0*
T0*
_output_shapes
: 2
	zeros/mul_
zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :è2
zeros/Less/yg

zeros/LessLesszeros/mul:z:0zeros/Less/y:output:0*
T0*
_output_shapes
: 2

zeros/Lessb
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :d2
zeros/packed/1
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros/packed_
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros/Constu
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd2
zeros`
zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B :d2
zeros_1/mul/yr
zeros_1/mulMulstrided_slice:output:0zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
zeros_1/mulc
zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :è2
zeros_1/Less/yo
zeros_1/LessLesszeros_1/mul:z:0zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
zeros_1/Lessf
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :d2
zeros_1/packed/1
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros_1/packedc
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros_1/Const}
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd2	
zeros_1u
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/permz
	transpose	Transposeinputstranspose/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
	transposeO
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:2	
Shape_1x
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2î
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2
TensorArrayV2/element_shape²
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2¿
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   27
5TensorArrayUnstack/TensorListFromTensor/element_shapeø
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02)
'TensorArrayUnstack/TensorListFromTensorx
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2ü
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
shrink_axis_mask2
strided_slice_2µ
"lstm_cell_71/MatMul/ReadVariableOpReadVariableOp+lstm_cell_71_matmul_readvariableop_resource*
_output_shapes
:	*
dtype02$
"lstm_cell_71/MatMul/ReadVariableOp­
lstm_cell_71/MatMulMatMulstrided_slice_2:output:0*lstm_cell_71/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell_71/MatMul»
$lstm_cell_71/MatMul_1/ReadVariableOpReadVariableOp-lstm_cell_71_matmul_1_readvariableop_resource*
_output_shapes
:	d*
dtype02&
$lstm_cell_71/MatMul_1/ReadVariableOp©
lstm_cell_71/MatMul_1MatMulzeros:output:0,lstm_cell_71/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell_71/MatMul_1 
lstm_cell_71/addAddV2lstm_cell_71/MatMul:product:0lstm_cell_71/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell_71/add´
#lstm_cell_71/BiasAdd/ReadVariableOpReadVariableOp,lstm_cell_71_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02%
#lstm_cell_71/BiasAdd/ReadVariableOp­
lstm_cell_71/BiasAddBiasAddlstm_cell_71/add:z:0+lstm_cell_71/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell_71/BiasAddj
lstm_cell_71/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_71/Const~
lstm_cell_71/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_71/split/split_dimó
lstm_cell_71/splitSplit%lstm_cell_71/split/split_dim:output:0lstm_cell_71/BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd*
	num_split2
lstm_cell_71/split
lstm_cell_71/SigmoidSigmoidlstm_cell_71/split:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd2
lstm_cell_71/Sigmoid
lstm_cell_71/Sigmoid_1Sigmoidlstm_cell_71/split:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd2
lstm_cell_71/Sigmoid_1
lstm_cell_71/mulMullstm_cell_71/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd2
lstm_cell_71/mul}
lstm_cell_71/ReluRelulstm_cell_71/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd2
lstm_cell_71/Relu
lstm_cell_71/mul_1Mullstm_cell_71/Sigmoid:y:0lstm_cell_71/Relu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd2
lstm_cell_71/mul_1
lstm_cell_71/add_1AddV2lstm_cell_71/mul:z:0lstm_cell_71/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd2
lstm_cell_71/add_1
lstm_cell_71/Sigmoid_2Sigmoidlstm_cell_71/split:output:3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd2
lstm_cell_71/Sigmoid_2|
lstm_cell_71/Relu_1Relulstm_cell_71/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd2
lstm_cell_71/Relu_1 
lstm_cell_71/mul_2Mullstm_cell_71/Sigmoid_2:y:0!lstm_cell_71/Relu_1:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd2
lstm_cell_71/mul_2
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿd   2
TensorArrayV2_1/element_shape¸
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2_1N
timeConst*
_output_shapes
: *
dtype0*
value	B : 2
time
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counterî
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0+lstm_cell_71_matmul_readvariableop_resource-lstm_cell_71_matmul_1_readvariableop_resource,lstm_cell_71_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd: : : : : *%
_read_only_resource_inputs
	
*
bodyR
while_body_624129*
condR
while_cond_624128*K
output_shapes:
8: : : : :ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd: : : : : *
parallel_iterations 2
whileµ
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿd   22
0TensorArrayV2Stack/TensorListStack/element_shapeè
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿd*
element_dtype02$
"TensorArrayV2Stack/TensorListStack
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿ2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd*
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/perm¥
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿd2
transpose_1f
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2	
runtimeæ
IdentityIdentitystrided_slice_3:output:0$^lstm_cell_71/BiasAdd/ReadVariableOp#^lstm_cell_71/MatMul/ReadVariableOp%^lstm_cell_71/MatMul_1/ReadVariableOp^while*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:ÿÿÿÿÿÿÿÿÿ:::2J
#lstm_cell_71/BiasAdd/ReadVariableOp#lstm_cell_71/BiasAdd/ReadVariableOp2H
"lstm_cell_71/MatMul/ReadVariableOp"lstm_cell_71/MatMul/ReadVariableOp2L
$lstm_cell_71/MatMul_1/ReadVariableOp$lstm_cell_71/MatMul_1/ReadVariableOp2
whilewhile:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
%

while_body_623827
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0
while_lstm_cell_71_623851_0
while_lstm_cell_71_623853_0
while_lstm_cell_71_623855_0
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor
while_lstm_cell_71_623851
while_lstm_cell_71_623853
while_lstm_cell_71_623855¢*while/lstm_cell_71/StatefulPartitionedCallÃ
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   29
7while/TensorArrayV2Read/TensorListGetItem/element_shapeÓ
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItemá
*while/lstm_cell_71/StatefulPartitionedCallStatefulPartitionedCall0while/TensorArrayV2Read/TensorListGetItem:item:0while_placeholder_2while_placeholder_3while_lstm_cell_71_623851_0while_lstm_cell_71_623853_0while_lstm_cell_71_623855_0*
Tin

2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *Q
fLRJ
H__inference_lstm_cell_71_layer_call_and_return_conditional_losses_6234012,
*while/lstm_cell_71/StatefulPartitionedCall÷
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder3while/lstm_cell_71/StatefulPartitionedCall:output:0*
_output_shapes
: *
element_dtype02,
*while/TensorArrayV2Write/TensorListSetItem\
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add/yi
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: 2
	while/add`
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add_1/yv
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: 2
while/add_1
while/IdentityIdentitywhile/add_1:z:0+^while/lstm_cell_71/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity
while/Identity_1Identitywhile_while_maximum_iterations+^while/lstm_cell_71/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity_1
while/Identity_2Identitywhile/add:z:0+^while/lstm_cell_71/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity_2º
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0+^while/lstm_cell_71/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity_3Ä
while/Identity_4Identity3while/lstm_cell_71/StatefulPartitionedCall:output:1+^while/lstm_cell_71/StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd2
while/Identity_4Ä
while/Identity_5Identity3while/lstm_cell_71/StatefulPartitionedCall:output:2+^while/lstm_cell_71/StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd2
while/Identity_5")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"8
while_lstm_cell_71_623851while_lstm_cell_71_623851_0"8
while_lstm_cell_71_623853while_lstm_cell_71_623853_0"8
while_lstm_cell_71_623855while_lstm_cell_71_623855_0"0
while_strided_slice_1while_strided_slice_1_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*Q
_input_shapes@
>: : : : :ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd: : :::2X
*while/lstm_cell_71/StatefulPartitionedCall*while/lstm_cell_71/StatefulPartitionedCall: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd:-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd:

_output_shapes
: :

_output_shapes
: "±L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*»
serving_default§
K
lstm_71_input:
serving_default_lstm_71_input:0ÿÿÿÿÿÿÿÿÿ<
dense_710
StatefulPartitionedCall:0ÿÿÿÿÿÿÿÿÿtensorflow/serving/predict:
"
layer_with_weights-0
layer-0
layer_with_weights-1
layer-1
	optimizer
	variables
regularization_losses
trainable_variables
	keras_api

signatures
E_default_save_signature
*F&call_and_return_all_conditional_losses
G__call__" 
_tf_keras_sequentialæ{"class_name": "Sequential", "name": "sequential_71", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "config": {"name": "sequential_71", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 20, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "lstm_71_input"}}, {"class_name": "LSTM", "config": {"name": "lstm_71", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 20, 1]}, "dtype": "float32", "return_sequences": false, "return_state": false, "go_backwards": false, "stateful": false, "unroll": false, "time_major": false, "units": 100, "activation": "relu", "recurrent_activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "recurrent_initializer": {"class_name": "Orthogonal", "config": {"gain": 1.0, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "unit_forget_bias": true, "kernel_regularizer": null, "recurrent_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "recurrent_constraint": null, "bias_constraint": null, "dropout": 0.0, "recurrent_dropout": 0.0, "implementation": 2}}, {"class_name": "Dense", "config": {"name": "dense_71", "trainable": true, "dtype": "float32", "units": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}]}, "input_spec": [{"class_name": "InputSpec", "config": {"dtype": null, "shape": {"class_name": "__tuple__", "items": [null, null, 1]}, "ndim": 3, "max_ndim": null, "min_ndim": null, "axes": {}}}], "build_input_shape": {"class_name": "TensorShape", "items": [null, 20, 1]}, "is_graph_network": true, "keras_version": "2.4.0", "backend": "tensorflow", "model_config": {"class_name": "Sequential", "config": {"name": "sequential_71", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 20, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "lstm_71_input"}}, {"class_name": "LSTM", "config": {"name": "lstm_71", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 20, 1]}, "dtype": "float32", "return_sequences": false, "return_state": false, "go_backwards": false, "stateful": false, "unroll": false, "time_major": false, "units": 100, "activation": "relu", "recurrent_activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "recurrent_initializer": {"class_name": "Orthogonal", "config": {"gain": 1.0, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "unit_forget_bias": true, "kernel_regularizer": null, "recurrent_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "recurrent_constraint": null, "bias_constraint": null, "dropout": 0.0, "recurrent_dropout": 0.0, "implementation": 2}}, {"class_name": "Dense", "config": {"name": "dense_71", "trainable": true, "dtype": "float32", "units": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}]}}, "training_config": {"loss": "mean_squared_error", "metrics": null, "weighted_metrics": null, "loss_weights": null, "optimizer_config": {"class_name": "Adam", "config": {"name": "Adam", "learning_rate": 0.0010000000474974513, "decay": 0.0, "beta_1": 0.8999999761581421, "beta_2": 0.9990000128746033, "epsilon": 1e-07, "amsgrad": false}}}}
·
	cell


state_spec
	variables
regularization_losses
trainable_variables
	keras_api
*H&call_and_return_all_conditional_losses
I__call__"
_tf_keras_rnn_layerð
{"class_name": "LSTM", "name": "lstm_71", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": {"class_name": "__tuple__", "items": [null, 20, 1]}, "stateful": false, "must_restore_from_config": false, "config": {"name": "lstm_71", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 20, 1]}, "dtype": "float32", "return_sequences": false, "return_state": false, "go_backwards": false, "stateful": false, "unroll": false, "time_major": false, "units": 100, "activation": "relu", "recurrent_activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "recurrent_initializer": {"class_name": "Orthogonal", "config": {"gain": 1.0, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "unit_forget_bias": true, "kernel_regularizer": null, "recurrent_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "recurrent_constraint": null, "bias_constraint": null, "dropout": 0.0, "recurrent_dropout": 0.0, "implementation": 2}, "input_spec": [{"class_name": "InputSpec", "config": {"dtype": null, "shape": {"class_name": "__tuple__", "items": [null, null, 1]}, "ndim": 3, "max_ndim": null, "min_ndim": null, "axes": {}}}], "build_input_shape": {"class_name": "TensorShape", "items": [null, 20, 1]}}
õ

kernel
bias
	variables
regularization_losses
trainable_variables
	keras_api
*J&call_and_return_all_conditional_losses
K__call__"Ð
_tf_keras_layer¶{"class_name": "Dense", "name": "dense_71", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_71", "trainable": true, "dtype": "float32", "units": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 100}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 100]}}
­
iter

beta_1

beta_2
	decay
learning_ratem;m<m=m>m?v@vAvBvCvD"
	optimizer
C
0
1
2
3
4"
trackable_list_wrapper
 "
trackable_list_wrapper
C
0
1
2
3
4"
trackable_list_wrapper
Ê
metrics
	variables
layer_regularization_losses
layer_metrics

 layers
!non_trainable_variables
regularization_losses
trainable_variables
G__call__
E_default_save_signature
*F&call_and_return_all_conditional_losses
&F"call_and_return_conditional_losses"
_generic_user_object
,
Lserving_default"
signature_map
«

kernel
recurrent_kernel
bias
"	variables
#regularization_losses
$trainable_variables
%	keras_api
*M&call_and_return_all_conditional_losses
N__call__"ð
_tf_keras_layerÖ{"class_name": "LSTMCell", "name": "lstm_cell_71", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "lstm_cell_71", "trainable": true, "dtype": "float32", "units": 100, "activation": "relu", "recurrent_activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "recurrent_initializer": {"class_name": "Orthogonal", "config": {"gain": 1.0, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "unit_forget_bias": true, "kernel_regularizer": null, "recurrent_regularizer": null, "bias_regularizer": null, "kernel_constraint": null, "recurrent_constraint": null, "bias_constraint": null, "dropout": 0.0, "recurrent_dropout": 0.0, "implementation": 2}}
 "
trackable_list_wrapper
5
0
1
2"
trackable_list_wrapper
 "
trackable_list_wrapper
5
0
1
2"
trackable_list_wrapper
¹
&metrics
	variables
'layer_regularization_losses
(layer_metrics

)states

*layers
+non_trainable_variables
regularization_losses
trainable_variables
I__call__
*H&call_and_return_all_conditional_losses
&H"call_and_return_conditional_losses"
_generic_user_object
!:d2dense_71/kernel
:2dense_71/bias
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
­
,metrics
	variables
-layer_regularization_losses
.layer_metrics

/layers
0non_trainable_variables
regularization_losses
trainable_variables
K__call__
*J&call_and_return_all_conditional_losses
&J"call_and_return_conditional_losses"
_generic_user_object
:	 (2	Adam/iter
: (2Adam/beta_1
: (2Adam/beta_2
: (2
Adam/decay
: (2Adam/learning_rate
.:,	2lstm_71/lstm_cell_71/kernel
8:6	d2%lstm_71/lstm_cell_71/recurrent_kernel
(:&2lstm_71/lstm_cell_71/bias
'
10"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
5
0
1
2"
trackable_list_wrapper
 "
trackable_list_wrapper
5
0
1
2"
trackable_list_wrapper
­
2metrics
"	variables
3layer_regularization_losses
4layer_metrics

5layers
6non_trainable_variables
#regularization_losses
$trainable_variables
N__call__
*M&call_and_return_all_conditional_losses
&M"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
'
	0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
»
	7total
	8count
9	variables
:	keras_api"
_tf_keras_metricj{"class_name": "Mean", "name": "loss", "dtype": "float32", "config": {"name": "loss", "dtype": "float32"}}
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
:  (2total
:  (2count
.
70
81"
trackable_list_wrapper
-
9	variables"
_generic_user_object
&:$d2Adam/dense_71/kernel/m
 :2Adam/dense_71/bias/m
3:1	2"Adam/lstm_71/lstm_cell_71/kernel/m
=:;	d2,Adam/lstm_71/lstm_cell_71/recurrent_kernel/m
-:+2 Adam/lstm_71/lstm_cell_71/bias/m
&:$d2Adam/dense_71/kernel/v
 :2Adam/dense_71/bias/v
3:1	2"Adam/lstm_71/lstm_cell_71/kernel/v
=:;	d2,Adam/lstm_71/lstm_cell_71/recurrent_kernel/v
-:+2 Adam/lstm_71/lstm_cell_71/bias/v
é2æ
!__inference__wrapped_model_623295À
²
FullArgSpec
args 
varargsjargs
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *0¢-
+(
lstm_71_inputÿÿÿÿÿÿÿÿÿ
ò2ï
I__inference_sequential_71_layer_call_and_return_conditional_losses_624287
I__inference_sequential_71_layer_call_and_return_conditional_losses_624534
I__inference_sequential_71_layer_call_and_return_conditional_losses_624271
I__inference_sequential_71_layer_call_and_return_conditional_losses_624693À
·²³
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
2
.__inference_sequential_71_layer_call_fn_624319
.__inference_sequential_71_layer_call_fn_624723
.__inference_sequential_71_layer_call_fn_624350
.__inference_sequential_71_layer_call_fn_624708À
·²³
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
ï2ì
C__inference_lstm_71_layer_call_and_return_conditional_losses_624876
C__inference_lstm_71_layer_call_and_return_conditional_losses_625029
C__inference_lstm_71_layer_call_and_return_conditional_losses_625357
C__inference_lstm_71_layer_call_and_return_conditional_losses_625204Õ
Ì²È
FullArgSpecB
args:7
jself
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaults

 
p 

 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
2
(__inference_lstm_71_layer_call_fn_625051
(__inference_lstm_71_layer_call_fn_625368
(__inference_lstm_71_layer_call_fn_625379
(__inference_lstm_71_layer_call_fn_625040Õ
Ì²È
FullArgSpecB
args:7
jself
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaults

 
p 

 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
î2ë
D__inference_dense_71_layer_call_and_return_conditional_losses_625389¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
Ó2Ð
)__inference_dense_71_layer_call_fn_625398¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ÑBÎ
$__inference_signature_wrapper_624375lstm_71_input"
²
FullArgSpec
args 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
Ø2Õ
H__inference_lstm_cell_71_layer_call_and_return_conditional_losses_625431
H__inference_lstm_cell_71_layer_call_and_return_conditional_losses_625464¾
µ²±
FullArgSpec3
args+(
jself
jinputs
jstates

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
¢2
-__inference_lstm_cell_71_layer_call_fn_625481
-__inference_lstm_cell_71_layer_call_fn_625498¾
µ²±
FullArgSpec3
args+(
jself
jinputs
jstates

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
!__inference__wrapped_model_623295x:¢7
0¢-
+(
lstm_71_inputÿÿÿÿÿÿÿÿÿ
ª "3ª0
.
dense_71"
dense_71ÿÿÿÿÿÿÿÿÿ¤
D__inference_dense_71_layer_call_and_return_conditional_losses_625389\/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿd
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 |
)__inference_dense_71_layer_call_fn_625398O/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿd
ª "ÿÿÿÿÿÿÿÿÿÄ
C__inference_lstm_71_layer_call_and_return_conditional_losses_624876}O¢L
E¢B
41
/,
inputs/0ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ

 
p

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿd
 Ä
C__inference_lstm_71_layer_call_and_return_conditional_losses_625029}O¢L
E¢B
41
/,
inputs/0ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ

 
p 

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿd
 ´
C__inference_lstm_71_layer_call_and_return_conditional_losses_625204m?¢<
5¢2
$!
inputsÿÿÿÿÿÿÿÿÿ

 
p

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿd
 ´
C__inference_lstm_71_layer_call_and_return_conditional_losses_625357m?¢<
5¢2
$!
inputsÿÿÿÿÿÿÿÿÿ

 
p 

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿd
 
(__inference_lstm_71_layer_call_fn_625040pO¢L
E¢B
41
/,
inputs/0ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ

 
p

 
ª "ÿÿÿÿÿÿÿÿÿd
(__inference_lstm_71_layer_call_fn_625051pO¢L
E¢B
41
/,
inputs/0ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ

 
p 

 
ª "ÿÿÿÿÿÿÿÿÿd
(__inference_lstm_71_layer_call_fn_625368`?¢<
5¢2
$!
inputsÿÿÿÿÿÿÿÿÿ

 
p

 
ª "ÿÿÿÿÿÿÿÿÿd
(__inference_lstm_71_layer_call_fn_625379`?¢<
5¢2
$!
inputsÿÿÿÿÿÿÿÿÿ

 
p 

 
ª "ÿÿÿÿÿÿÿÿÿdÊ
H__inference_lstm_cell_71_layer_call_and_return_conditional_losses_625431ý¢}
v¢s
 
inputsÿÿÿÿÿÿÿÿÿ
K¢H
"
states/0ÿÿÿÿÿÿÿÿÿd
"
states/1ÿÿÿÿÿÿÿÿÿd
p
ª "s¢p
i¢f

0/0ÿÿÿÿÿÿÿÿÿd
EB

0/1/0ÿÿÿÿÿÿÿÿÿd

0/1/1ÿÿÿÿÿÿÿÿÿd
 Ê
H__inference_lstm_cell_71_layer_call_and_return_conditional_losses_625464ý¢}
v¢s
 
inputsÿÿÿÿÿÿÿÿÿ
K¢H
"
states/0ÿÿÿÿÿÿÿÿÿd
"
states/1ÿÿÿÿÿÿÿÿÿd
p 
ª "s¢p
i¢f

0/0ÿÿÿÿÿÿÿÿÿd
EB

0/1/0ÿÿÿÿÿÿÿÿÿd

0/1/1ÿÿÿÿÿÿÿÿÿd
 
-__inference_lstm_cell_71_layer_call_fn_625481í¢}
v¢s
 
inputsÿÿÿÿÿÿÿÿÿ
K¢H
"
states/0ÿÿÿÿÿÿÿÿÿd
"
states/1ÿÿÿÿÿÿÿÿÿd
p
ª "c¢`

0ÿÿÿÿÿÿÿÿÿd
A>

1/0ÿÿÿÿÿÿÿÿÿd

1/1ÿÿÿÿÿÿÿÿÿd
-__inference_lstm_cell_71_layer_call_fn_625498í¢}
v¢s
 
inputsÿÿÿÿÿÿÿÿÿ
K¢H
"
states/0ÿÿÿÿÿÿÿÿÿd
"
states/1ÿÿÿÿÿÿÿÿÿd
p 
ª "c¢`

0ÿÿÿÿÿÿÿÿÿd
A>

1/0ÿÿÿÿÿÿÿÿÿd

1/1ÿÿÿÿÿÿÿÿÿd¿
I__inference_sequential_71_layer_call_and_return_conditional_losses_624271rB¢?
8¢5
+(
lstm_71_inputÿÿÿÿÿÿÿÿÿ
p

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 ¿
I__inference_sequential_71_layer_call_and_return_conditional_losses_624287rB¢?
8¢5
+(
lstm_71_inputÿÿÿÿÿÿÿÿÿ
p 

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 ¸
I__inference_sequential_71_layer_call_and_return_conditional_losses_624534k;¢8
1¢.
$!
inputsÿÿÿÿÿÿÿÿÿ
p

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 ¸
I__inference_sequential_71_layer_call_and_return_conditional_losses_624693k;¢8
1¢.
$!
inputsÿÿÿÿÿÿÿÿÿ
p 

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 
.__inference_sequential_71_layer_call_fn_624319eB¢?
8¢5
+(
lstm_71_inputÿÿÿÿÿÿÿÿÿ
p

 
ª "ÿÿÿÿÿÿÿÿÿ
.__inference_sequential_71_layer_call_fn_624350eB¢?
8¢5
+(
lstm_71_inputÿÿÿÿÿÿÿÿÿ
p 

 
ª "ÿÿÿÿÿÿÿÿÿ
.__inference_sequential_71_layer_call_fn_624708^;¢8
1¢.
$!
inputsÿÿÿÿÿÿÿÿÿ
p

 
ª "ÿÿÿÿÿÿÿÿÿ
.__inference_sequential_71_layer_call_fn_624723^;¢8
1¢.
$!
inputsÿÿÿÿÿÿÿÿÿ
p 

 
ª "ÿÿÿÿÿÿÿÿÿ²
$__inference_signature_wrapper_624375K¢H
¢ 
Aª>
<
lstm_71_input+(
lstm_71_inputÿÿÿÿÿÿÿÿÿ"3ª0
.
dense_71"
dense_71ÿÿÿÿÿÿÿÿÿ