??
??
B
AddV2
x"T
y"T
z"T"
Ttype:
2	??
B
AssignVariableOp
resource
value"dtype"
dtypetype?
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
delete_old_dirsbool(?
=
Mul
x"T
y"T
z"T"
Ttype:
2	?
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
dtypetype?
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
list(type)(0?
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0?
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
?
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
executor_typestring ?
@
StaticRegexFullMatch	
input

output
"
patternstring
?
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
?
TensorListFromTensor
tensor"element_dtype
element_shape"
shape_type
output_handle"
element_dtypetype"

shape_typetype:
2	
?
TensorListReserve
element_shape"
shape_type
num_elements

handle"
element_dtypetype"

shape_typetype:
2	
?
TensorListStack
input_handle
element_shape
tensor"element_dtype"
element_dtypetype" 
num_elementsint?????????
P
	Transpose
x"T
perm"Tperm
y"T"	
Ttype"
Tpermtype0:
2	
?
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 ?
?
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
?"serve*2.4.12v2.4.1-0-g85c8b2a817f8??
|
dense_128/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:2*!
shared_namedense_128/kernel
u
$dense_128/kernel/Read/ReadVariableOpReadVariableOpdense_128/kernel*
_output_shapes

:2*
dtype0
t
dense_128/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_128/bias
m
"dense_128/bias/Read/ReadVariableOpReadVariableOpdense_128/bias*
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
?
lstm_134/lstm_cell_134/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?*.
shared_namelstm_134/lstm_cell_134/kernel
?
1lstm_134/lstm_cell_134/kernel/Read/ReadVariableOpReadVariableOplstm_134/lstm_cell_134/kernel*
_output_shapes
:	?*
dtype0
?
'lstm_134/lstm_cell_134/recurrent_kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	2?*8
shared_name)'lstm_134/lstm_cell_134/recurrent_kernel
?
;lstm_134/lstm_cell_134/recurrent_kernel/Read/ReadVariableOpReadVariableOp'lstm_134/lstm_cell_134/recurrent_kernel*
_output_shapes
:	2?*
dtype0
?
lstm_134/lstm_cell_134/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*,
shared_namelstm_134/lstm_cell_134/bias
?
/lstm_134/lstm_cell_134/bias/Read/ReadVariableOpReadVariableOplstm_134/lstm_cell_134/bias*
_output_shapes	
:?*
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
?
Adam/dense_128/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:2*(
shared_nameAdam/dense_128/kernel/m
?
+Adam/dense_128/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_128/kernel/m*
_output_shapes

:2*
dtype0
?
Adam/dense_128/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*&
shared_nameAdam/dense_128/bias/m
{
)Adam/dense_128/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_128/bias/m*
_output_shapes
:*
dtype0
?
$Adam/lstm_134/lstm_cell_134/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?*5
shared_name&$Adam/lstm_134/lstm_cell_134/kernel/m
?
8Adam/lstm_134/lstm_cell_134/kernel/m/Read/ReadVariableOpReadVariableOp$Adam/lstm_134/lstm_cell_134/kernel/m*
_output_shapes
:	?*
dtype0
?
.Adam/lstm_134/lstm_cell_134/recurrent_kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	2?*?
shared_name0.Adam/lstm_134/lstm_cell_134/recurrent_kernel/m
?
BAdam/lstm_134/lstm_cell_134/recurrent_kernel/m/Read/ReadVariableOpReadVariableOp.Adam/lstm_134/lstm_cell_134/recurrent_kernel/m*
_output_shapes
:	2?*
dtype0
?
"Adam/lstm_134/lstm_cell_134/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*3
shared_name$"Adam/lstm_134/lstm_cell_134/bias/m
?
6Adam/lstm_134/lstm_cell_134/bias/m/Read/ReadVariableOpReadVariableOp"Adam/lstm_134/lstm_cell_134/bias/m*
_output_shapes	
:?*
dtype0
?
Adam/dense_128/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:2*(
shared_nameAdam/dense_128/kernel/v
?
+Adam/dense_128/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_128/kernel/v*
_output_shapes

:2*
dtype0
?
Adam/dense_128/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*&
shared_nameAdam/dense_128/bias/v
{
)Adam/dense_128/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_128/bias/v*
_output_shapes
:*
dtype0
?
$Adam/lstm_134/lstm_cell_134/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?*5
shared_name&$Adam/lstm_134/lstm_cell_134/kernel/v
?
8Adam/lstm_134/lstm_cell_134/kernel/v/Read/ReadVariableOpReadVariableOp$Adam/lstm_134/lstm_cell_134/kernel/v*
_output_shapes
:	?*
dtype0
?
.Adam/lstm_134/lstm_cell_134/recurrent_kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	2?*?
shared_name0.Adam/lstm_134/lstm_cell_134/recurrent_kernel/v
?
BAdam/lstm_134/lstm_cell_134/recurrent_kernel/v/Read/ReadVariableOpReadVariableOp.Adam/lstm_134/lstm_cell_134/recurrent_kernel/v*
_output_shapes
:	2?*
dtype0
?
"Adam/lstm_134/lstm_cell_134/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*3
shared_name$"Adam/lstm_134/lstm_cell_134/bias/v
?
6Adam/lstm_134/lstm_cell_134/bias/v/Read/ReadVariableOpReadVariableOp"Adam/lstm_134/lstm_cell_134/bias/v*
_output_shapes	
:?*
dtype0

NoOpNoOp
? 
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*? 
value? B? B?
?
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
?
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
?
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
?
&metrics
	variables
'layer_regularization_losses
(layer_metrics

)states

*layers
+non_trainable_variables
regularization_losses
trainable_variables
\Z
VARIABLE_VALUEdense_128/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEdense_128/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE

0
1
 

0
1
?
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
YW
VARIABLE_VALUElstm_134/lstm_cell_134/kernel&variables/0/.ATTRIBUTES/VARIABLE_VALUE
ca
VARIABLE_VALUE'lstm_134/lstm_cell_134/recurrent_kernel&variables/1/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUElstm_134/lstm_cell_134/bias&variables/2/.ATTRIBUTES/VARIABLE_VALUE
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
?
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
}
VARIABLE_VALUEAdam/dense_128/kernel/mRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/dense_128/bias/mPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUE$Adam/lstm_134/lstm_cell_134/kernel/mBvariables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE.Adam/lstm_134/lstm_cell_134/recurrent_kernel/mBvariables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUE"Adam/lstm_134/lstm_cell_134/bias/mBvariables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/dense_128/kernel/vRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/dense_128/bias/vPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUE$Adam/lstm_134/lstm_cell_134/kernel/vBvariables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE.Adam/lstm_134/lstm_cell_134/recurrent_kernel/vBvariables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUE"Adam/lstm_134/lstm_cell_134/bias/vBvariables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
?
serving_default_lstm_134_inputPlaceholder*+
_output_shapes
:?????????*
dtype0* 
shape:?????????
?
StatefulPartitionedCallStatefulPartitionedCallserving_default_lstm_134_inputlstm_134/lstm_cell_134/kernel'lstm_134/lstm_cell_134/recurrent_kernellstm_134/lstm_cell_134/biasdense_128/kerneldense_128/bias*
Tin

2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*'
_read_only_resource_inputs	
*-
config_proto

CPU

GPU 2J 8? *.
f)R'
%__inference_signature_wrapper_1064666
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
?

StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename$dense_128/kernel/Read/ReadVariableOp"dense_128/bias/Read/ReadVariableOpAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOp&Adam/learning_rate/Read/ReadVariableOp1lstm_134/lstm_cell_134/kernel/Read/ReadVariableOp;lstm_134/lstm_cell_134/recurrent_kernel/Read/ReadVariableOp/lstm_134/lstm_cell_134/bias/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOp+Adam/dense_128/kernel/m/Read/ReadVariableOp)Adam/dense_128/bias/m/Read/ReadVariableOp8Adam/lstm_134/lstm_cell_134/kernel/m/Read/ReadVariableOpBAdam/lstm_134/lstm_cell_134/recurrent_kernel/m/Read/ReadVariableOp6Adam/lstm_134/lstm_cell_134/bias/m/Read/ReadVariableOp+Adam/dense_128/kernel/v/Read/ReadVariableOp)Adam/dense_128/bias/v/Read/ReadVariableOp8Adam/lstm_134/lstm_cell_134/kernel/v/Read/ReadVariableOpBAdam/lstm_134/lstm_cell_134/recurrent_kernel/v/Read/ReadVariableOp6Adam/lstm_134/lstm_cell_134/bias/v/Read/ReadVariableOpConst*#
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
GPU 2J 8? *)
f$R"
 __inference__traced_save_1065878
?
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamedense_128/kerneldense_128/bias	Adam/iterAdam/beta_1Adam/beta_2
Adam/decayAdam/learning_ratelstm_134/lstm_cell_134/kernel'lstm_134/lstm_cell_134/recurrent_kernellstm_134/lstm_cell_134/biastotalcountAdam/dense_128/kernel/mAdam/dense_128/bias/m$Adam/lstm_134/lstm_cell_134/kernel/m.Adam/lstm_134/lstm_cell_134/recurrent_kernel/m"Adam/lstm_134/lstm_cell_134/bias/mAdam/dense_128/kernel/vAdam/dense_128/bias/v$Adam/lstm_134/lstm_cell_134/kernel/v.Adam/lstm_134/lstm_cell_134/recurrent_kernel/v"Adam/lstm_134/lstm_cell_134/bias/v*"
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
GPU 2J 8? *,
f'R%
#__inference__traced_restore_1065954??
?\
?
E__inference_lstm_134_layer_call_and_return_conditional_losses_1065167
inputs_00
,lstm_cell_134_matmul_readvariableop_resource2
.lstm_cell_134_matmul_1_readvariableop_resource1
-lstm_cell_134_biasadd_readvariableop_resource
identity??$lstm_cell_134/BiasAdd/ReadVariableOp?#lstm_cell_134/MatMul/ReadVariableOp?%lstm_cell_134/MatMul_1/ReadVariableOp?whileF
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
strided_slice/stack_2?
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
value	B :22
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
B :?2
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
value	B :22
zeros/packed/1?
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
:?????????22
zeros`
zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B :22
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
B :?2
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
value	B :22
zeros_1/packed/1?
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
:?????????22	
zeros_1u
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/perm?
	transpose	Transposeinputs_0transpose/perm:output:0*
T0*4
_output_shapes"
 :??????????????????2
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
strided_slice_1/stack_2?
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1?
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
TensorArrayV2/element_shape?
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2?
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   27
5TensorArrayUnstack/TensorListFromTensor/element_shape?
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
strided_slice_2/stack_2?
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*
shrink_axis_mask2
strided_slice_2?
#lstm_cell_134/MatMul/ReadVariableOpReadVariableOp,lstm_cell_134_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype02%
#lstm_cell_134/MatMul/ReadVariableOp?
lstm_cell_134/MatMulMatMulstrided_slice_2:output:0+lstm_cell_134/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
lstm_cell_134/MatMul?
%lstm_cell_134/MatMul_1/ReadVariableOpReadVariableOp.lstm_cell_134_matmul_1_readvariableop_resource*
_output_shapes
:	2?*
dtype02'
%lstm_cell_134/MatMul_1/ReadVariableOp?
lstm_cell_134/MatMul_1MatMulzeros:output:0-lstm_cell_134/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
lstm_cell_134/MatMul_1?
lstm_cell_134/addAddV2lstm_cell_134/MatMul:product:0 lstm_cell_134/MatMul_1:product:0*
T0*(
_output_shapes
:??????????2
lstm_cell_134/add?
$lstm_cell_134/BiasAdd/ReadVariableOpReadVariableOp-lstm_cell_134_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02&
$lstm_cell_134/BiasAdd/ReadVariableOp?
lstm_cell_134/BiasAddBiasAddlstm_cell_134/add:z:0,lstm_cell_134/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
lstm_cell_134/BiasAddl
lstm_cell_134/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_134/Const?
lstm_cell_134/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_134/split/split_dim?
lstm_cell_134/splitSplit&lstm_cell_134/split/split_dim:output:0lstm_cell_134/BiasAdd:output:0*
T0*`
_output_shapesN
L:?????????2:?????????2:?????????2:?????????2*
	num_split2
lstm_cell_134/split?
lstm_cell_134/SigmoidSigmoidlstm_cell_134/split:output:0*
T0*'
_output_shapes
:?????????22
lstm_cell_134/Sigmoid?
lstm_cell_134/Sigmoid_1Sigmoidlstm_cell_134/split:output:1*
T0*'
_output_shapes
:?????????22
lstm_cell_134/Sigmoid_1?
lstm_cell_134/mulMullstm_cell_134/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:?????????22
lstm_cell_134/mul?
lstm_cell_134/ReluRelulstm_cell_134/split:output:2*
T0*'
_output_shapes
:?????????22
lstm_cell_134/Relu?
lstm_cell_134/mul_1Mullstm_cell_134/Sigmoid:y:0 lstm_cell_134/Relu:activations:0*
T0*'
_output_shapes
:?????????22
lstm_cell_134/mul_1?
lstm_cell_134/add_1AddV2lstm_cell_134/mul:z:0lstm_cell_134/mul_1:z:0*
T0*'
_output_shapes
:?????????22
lstm_cell_134/add_1?
lstm_cell_134/Sigmoid_2Sigmoidlstm_cell_134/split:output:3*
T0*'
_output_shapes
:?????????22
lstm_cell_134/Sigmoid_2
lstm_cell_134/Relu_1Relulstm_cell_134/add_1:z:0*
T0*'
_output_shapes
:?????????22
lstm_cell_134/Relu_1?
lstm_cell_134/mul_2Mullstm_cell_134/Sigmoid_2:y:0"lstm_cell_134/Relu_1:activations:0*
T0*'
_output_shapes
:?????????22
lstm_cell_134/mul_2?
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????2   2
TensorArrayV2_1/element_shape?
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
?????????2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counter?
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0,lstm_cell_134_matmul_readvariableop_resource.lstm_cell_134_matmul_1_readvariableop_resource-lstm_cell_134_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :?????????2:?????????2: : : : : *%
_read_only_resource_inputs
	
*
bodyR
while_body_1065082*
condR
while_cond_1065081*K
output_shapes:
8: : : : :?????????2:?????????2: : : : : *
parallel_iterations 2
while?
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????2   22
0TensorArrayV2Stack/TensorListStack/element_shape?
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :??????????????????2*
element_dtype02$
"TensorArrayV2Stack/TensorListStack?
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????2
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
strided_slice_3/stack_2?
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????2*
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/perm?
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*4
_output_shapes"
 :??????????????????22
transpose_1f
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2	
runtime?
IdentityIdentitystrided_slice_3:output:0%^lstm_cell_134/BiasAdd/ReadVariableOp$^lstm_cell_134/MatMul/ReadVariableOp&^lstm_cell_134/MatMul_1/ReadVariableOp^while*
T0*'
_output_shapes
:?????????22

Identity"
identityIdentity:output:0*?
_input_shapes.
,:??????????????????:::2L
$lstm_cell_134/BiasAdd/ReadVariableOp$lstm_cell_134/BiasAdd/ReadVariableOp2J
#lstm_cell_134/MatMul/ReadVariableOp#lstm_cell_134/MatMul/ReadVariableOp2N
%lstm_cell_134/MatMul_1/ReadVariableOp%lstm_cell_134/MatMul_1/ReadVariableOp2
whilewhile:^ Z
4
_output_shapes"
 :??????????????????
"
_user_specified_name
inputs/0
?

?
lstm_134_while_cond_1064892.
*lstm_134_while_lstm_134_while_loop_counter4
0lstm_134_while_lstm_134_while_maximum_iterations
lstm_134_while_placeholder 
lstm_134_while_placeholder_1 
lstm_134_while_placeholder_2 
lstm_134_while_placeholder_30
,lstm_134_while_less_lstm_134_strided_slice_1G
Clstm_134_while_lstm_134_while_cond_1064892___redundant_placeholder0G
Clstm_134_while_lstm_134_while_cond_1064892___redundant_placeholder1G
Clstm_134_while_lstm_134_while_cond_1064892___redundant_placeholder2G
Clstm_134_while_lstm_134_while_cond_1064892___redundant_placeholder3
lstm_134_while_identity
?
lstm_134/while/LessLesslstm_134_while_placeholder,lstm_134_while_less_lstm_134_strided_slice_1*
T0*
_output_shapes
: 2
lstm_134/while/Lessx
lstm_134/while/IdentityIdentitylstm_134/while/Less:z:0*
T0
*
_output_shapes
: 2
lstm_134/while/Identity";
lstm_134_while_identity lstm_134/while/Identity:output:0*S
_input_shapesB
@: : : : :?????????2:?????????2: ::::: 
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
:?????????2:-)
'
_output_shapes
:?????????2:

_output_shapes
: :

_output_shapes
:
?D
?
E__inference_lstm_134_layer_call_and_return_conditional_losses_1064055

inputs
lstm_cell_134_1063973
lstm_cell_134_1063975
lstm_cell_134_1063977
identity??%lstm_cell_134/StatefulPartitionedCall?whileD
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
strided_slice/stack_2?
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
value	B :22
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
B :?2
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
value	B :22
zeros/packed/1?
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
:?????????22
zeros`
zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B :22
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
B :?2
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
value	B :22
zeros_1/packed/1?
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
:?????????22	
zeros_1u
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/perm?
	transpose	Transposeinputstranspose/perm:output:0*
T0*4
_output_shapes"
 :??????????????????2
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
strided_slice_1/stack_2?
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1?
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
TensorArrayV2/element_shape?
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2?
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   27
5TensorArrayUnstack/TensorListFromTensor/element_shape?
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
strided_slice_2/stack_2?
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*
shrink_axis_mask2
strided_slice_2?
%lstm_cell_134/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_2:output:0zeros:output:0zeros_1:output:0lstm_cell_134_1063973lstm_cell_134_1063975lstm_cell_134_1063977*
Tin

2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:?????????2:?????????2:?????????2*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *S
fNRL
J__inference_lstm_cell_134_layer_call_and_return_conditional_losses_10636592'
%lstm_cell_134/StatefulPartitionedCall?
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????2   2
TensorArrayV2_1/element_shape?
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
?????????2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counter?
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0lstm_cell_134_1063973lstm_cell_134_1063975lstm_cell_134_1063977*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :?????????2:?????????2: : : : : *%
_read_only_resource_inputs
	
*
bodyR
while_body_1063986*
condR
while_cond_1063985*K
output_shapes:
8: : : : :?????????2:?????????2: : : : : *
parallel_iterations 2
while?
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????2   22
0TensorArrayV2Stack/TensorListStack/element_shape?
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :??????????????????2*
element_dtype02$
"TensorArrayV2Stack/TensorListStack?
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????2
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
strided_slice_3/stack_2?
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????2*
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/perm?
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*4
_output_shapes"
 :??????????????????22
transpose_1f
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2	
runtime?
IdentityIdentitystrided_slice_3:output:0&^lstm_cell_134/StatefulPartitionedCall^while*
T0*'
_output_shapes
:?????????22

Identity"
identityIdentity:output:0*?
_input_shapes.
,:??????????????????:::2N
%lstm_cell_134/StatefulPartitionedCall%lstm_cell_134/StatefulPartitionedCall2
whilewhile:\ X
4
_output_shapes"
 :??????????????????
 
_user_specified_nameinputs
ݔ
?
"__inference__wrapped_model_1063586
lstm_134_inputH
Dsequential_129_lstm_134_lstm_cell_134_matmul_readvariableop_resourceJ
Fsequential_129_lstm_134_lstm_cell_134_matmul_1_readvariableop_resourceI
Esequential_129_lstm_134_lstm_cell_134_biasadd_readvariableop_resource;
7sequential_129_dense_128_matmul_readvariableop_resource<
8sequential_129_dense_128_biasadd_readvariableop_resource
identity??/sequential_129/dense_128/BiasAdd/ReadVariableOp?.sequential_129/dense_128/MatMul/ReadVariableOp?<sequential_129/lstm_134/lstm_cell_134/BiasAdd/ReadVariableOp?;sequential_129/lstm_134/lstm_cell_134/MatMul/ReadVariableOp?=sequential_129/lstm_134/lstm_cell_134/MatMul_1/ReadVariableOp?sequential_129/lstm_134/while|
sequential_129/lstm_134/ShapeShapelstm_134_input*
T0*
_output_shapes
:2
sequential_129/lstm_134/Shape?
+sequential_129/lstm_134/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2-
+sequential_129/lstm_134/strided_slice/stack?
-sequential_129/lstm_134/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2/
-sequential_129/lstm_134/strided_slice/stack_1?
-sequential_129/lstm_134/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2/
-sequential_129/lstm_134/strided_slice/stack_2?
%sequential_129/lstm_134/strided_sliceStridedSlice&sequential_129/lstm_134/Shape:output:04sequential_129/lstm_134/strided_slice/stack:output:06sequential_129/lstm_134/strided_slice/stack_1:output:06sequential_129/lstm_134/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2'
%sequential_129/lstm_134/strided_slice?
#sequential_129/lstm_134/zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :22%
#sequential_129/lstm_134/zeros/mul/y?
!sequential_129/lstm_134/zeros/mulMul.sequential_129/lstm_134/strided_slice:output:0,sequential_129/lstm_134/zeros/mul/y:output:0*
T0*
_output_shapes
: 2#
!sequential_129/lstm_134/zeros/mul?
$sequential_129/lstm_134/zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2&
$sequential_129/lstm_134/zeros/Less/y?
"sequential_129/lstm_134/zeros/LessLess%sequential_129/lstm_134/zeros/mul:z:0-sequential_129/lstm_134/zeros/Less/y:output:0*
T0*
_output_shapes
: 2$
"sequential_129/lstm_134/zeros/Less?
&sequential_129/lstm_134/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :22(
&sequential_129/lstm_134/zeros/packed/1?
$sequential_129/lstm_134/zeros/packedPack.sequential_129/lstm_134/strided_slice:output:0/sequential_129/lstm_134/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2&
$sequential_129/lstm_134/zeros/packed?
#sequential_129/lstm_134/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2%
#sequential_129/lstm_134/zeros/Const?
sequential_129/lstm_134/zerosFill-sequential_129/lstm_134/zeros/packed:output:0,sequential_129/lstm_134/zeros/Const:output:0*
T0*'
_output_shapes
:?????????22
sequential_129/lstm_134/zeros?
%sequential_129/lstm_134/zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B :22'
%sequential_129/lstm_134/zeros_1/mul/y?
#sequential_129/lstm_134/zeros_1/mulMul.sequential_129/lstm_134/strided_slice:output:0.sequential_129/lstm_134/zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2%
#sequential_129/lstm_134/zeros_1/mul?
&sequential_129/lstm_134/zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2(
&sequential_129/lstm_134/zeros_1/Less/y?
$sequential_129/lstm_134/zeros_1/LessLess'sequential_129/lstm_134/zeros_1/mul:z:0/sequential_129/lstm_134/zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2&
$sequential_129/lstm_134/zeros_1/Less?
(sequential_129/lstm_134/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :22*
(sequential_129/lstm_134/zeros_1/packed/1?
&sequential_129/lstm_134/zeros_1/packedPack.sequential_129/lstm_134/strided_slice:output:01sequential_129/lstm_134/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2(
&sequential_129/lstm_134/zeros_1/packed?
%sequential_129/lstm_134/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2'
%sequential_129/lstm_134/zeros_1/Const?
sequential_129/lstm_134/zeros_1Fill/sequential_129/lstm_134/zeros_1/packed:output:0.sequential_129/lstm_134/zeros_1/Const:output:0*
T0*'
_output_shapes
:?????????22!
sequential_129/lstm_134/zeros_1?
&sequential_129/lstm_134/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2(
&sequential_129/lstm_134/transpose/perm?
!sequential_129/lstm_134/transpose	Transposelstm_134_input/sequential_129/lstm_134/transpose/perm:output:0*
T0*+
_output_shapes
:?????????2#
!sequential_129/lstm_134/transpose?
sequential_129/lstm_134/Shape_1Shape%sequential_129/lstm_134/transpose:y:0*
T0*
_output_shapes
:2!
sequential_129/lstm_134/Shape_1?
-sequential_129/lstm_134/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2/
-sequential_129/lstm_134/strided_slice_1/stack?
/sequential_129/lstm_134/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:21
/sequential_129/lstm_134/strided_slice_1/stack_1?
/sequential_129/lstm_134/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:21
/sequential_129/lstm_134/strided_slice_1/stack_2?
'sequential_129/lstm_134/strided_slice_1StridedSlice(sequential_129/lstm_134/Shape_1:output:06sequential_129/lstm_134/strided_slice_1/stack:output:08sequential_129/lstm_134/strided_slice_1/stack_1:output:08sequential_129/lstm_134/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2)
'sequential_129/lstm_134/strided_slice_1?
3sequential_129/lstm_134/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
?????????25
3sequential_129/lstm_134/TensorArrayV2/element_shape?
%sequential_129/lstm_134/TensorArrayV2TensorListReserve<sequential_129/lstm_134/TensorArrayV2/element_shape:output:00sequential_129/lstm_134/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02'
%sequential_129/lstm_134/TensorArrayV2?
Msequential_129/lstm_134/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   2O
Msequential_129/lstm_134/TensorArrayUnstack/TensorListFromTensor/element_shape?
?sequential_129/lstm_134/TensorArrayUnstack/TensorListFromTensorTensorListFromTensor%sequential_129/lstm_134/transpose:y:0Vsequential_129/lstm_134/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02A
?sequential_129/lstm_134/TensorArrayUnstack/TensorListFromTensor?
-sequential_129/lstm_134/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2/
-sequential_129/lstm_134/strided_slice_2/stack?
/sequential_129/lstm_134/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:21
/sequential_129/lstm_134/strided_slice_2/stack_1?
/sequential_129/lstm_134/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:21
/sequential_129/lstm_134/strided_slice_2/stack_2?
'sequential_129/lstm_134/strided_slice_2StridedSlice%sequential_129/lstm_134/transpose:y:06sequential_129/lstm_134/strided_slice_2/stack:output:08sequential_129/lstm_134/strided_slice_2/stack_1:output:08sequential_129/lstm_134/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*
shrink_axis_mask2)
'sequential_129/lstm_134/strided_slice_2?
;sequential_129/lstm_134/lstm_cell_134/MatMul/ReadVariableOpReadVariableOpDsequential_129_lstm_134_lstm_cell_134_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype02=
;sequential_129/lstm_134/lstm_cell_134/MatMul/ReadVariableOp?
,sequential_129/lstm_134/lstm_cell_134/MatMulMatMul0sequential_129/lstm_134/strided_slice_2:output:0Csequential_129/lstm_134/lstm_cell_134/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2.
,sequential_129/lstm_134/lstm_cell_134/MatMul?
=sequential_129/lstm_134/lstm_cell_134/MatMul_1/ReadVariableOpReadVariableOpFsequential_129_lstm_134_lstm_cell_134_matmul_1_readvariableop_resource*
_output_shapes
:	2?*
dtype02?
=sequential_129/lstm_134/lstm_cell_134/MatMul_1/ReadVariableOp?
.sequential_129/lstm_134/lstm_cell_134/MatMul_1MatMul&sequential_129/lstm_134/zeros:output:0Esequential_129/lstm_134/lstm_cell_134/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????20
.sequential_129/lstm_134/lstm_cell_134/MatMul_1?
)sequential_129/lstm_134/lstm_cell_134/addAddV26sequential_129/lstm_134/lstm_cell_134/MatMul:product:08sequential_129/lstm_134/lstm_cell_134/MatMul_1:product:0*
T0*(
_output_shapes
:??????????2+
)sequential_129/lstm_134/lstm_cell_134/add?
<sequential_129/lstm_134/lstm_cell_134/BiasAdd/ReadVariableOpReadVariableOpEsequential_129_lstm_134_lstm_cell_134_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02>
<sequential_129/lstm_134/lstm_cell_134/BiasAdd/ReadVariableOp?
-sequential_129/lstm_134/lstm_cell_134/BiasAddBiasAdd-sequential_129/lstm_134/lstm_cell_134/add:z:0Dsequential_129/lstm_134/lstm_cell_134/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2/
-sequential_129/lstm_134/lstm_cell_134/BiasAdd?
+sequential_129/lstm_134/lstm_cell_134/ConstConst*
_output_shapes
: *
dtype0*
value	B :2-
+sequential_129/lstm_134/lstm_cell_134/Const?
5sequential_129/lstm_134/lstm_cell_134/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :27
5sequential_129/lstm_134/lstm_cell_134/split/split_dim?
+sequential_129/lstm_134/lstm_cell_134/splitSplit>sequential_129/lstm_134/lstm_cell_134/split/split_dim:output:06sequential_129/lstm_134/lstm_cell_134/BiasAdd:output:0*
T0*`
_output_shapesN
L:?????????2:?????????2:?????????2:?????????2*
	num_split2-
+sequential_129/lstm_134/lstm_cell_134/split?
-sequential_129/lstm_134/lstm_cell_134/SigmoidSigmoid4sequential_129/lstm_134/lstm_cell_134/split:output:0*
T0*'
_output_shapes
:?????????22/
-sequential_129/lstm_134/lstm_cell_134/Sigmoid?
/sequential_129/lstm_134/lstm_cell_134/Sigmoid_1Sigmoid4sequential_129/lstm_134/lstm_cell_134/split:output:1*
T0*'
_output_shapes
:?????????221
/sequential_129/lstm_134/lstm_cell_134/Sigmoid_1?
)sequential_129/lstm_134/lstm_cell_134/mulMul3sequential_129/lstm_134/lstm_cell_134/Sigmoid_1:y:0(sequential_129/lstm_134/zeros_1:output:0*
T0*'
_output_shapes
:?????????22+
)sequential_129/lstm_134/lstm_cell_134/mul?
*sequential_129/lstm_134/lstm_cell_134/ReluRelu4sequential_129/lstm_134/lstm_cell_134/split:output:2*
T0*'
_output_shapes
:?????????22,
*sequential_129/lstm_134/lstm_cell_134/Relu?
+sequential_129/lstm_134/lstm_cell_134/mul_1Mul1sequential_129/lstm_134/lstm_cell_134/Sigmoid:y:08sequential_129/lstm_134/lstm_cell_134/Relu:activations:0*
T0*'
_output_shapes
:?????????22-
+sequential_129/lstm_134/lstm_cell_134/mul_1?
+sequential_129/lstm_134/lstm_cell_134/add_1AddV2-sequential_129/lstm_134/lstm_cell_134/mul:z:0/sequential_129/lstm_134/lstm_cell_134/mul_1:z:0*
T0*'
_output_shapes
:?????????22-
+sequential_129/lstm_134/lstm_cell_134/add_1?
/sequential_129/lstm_134/lstm_cell_134/Sigmoid_2Sigmoid4sequential_129/lstm_134/lstm_cell_134/split:output:3*
T0*'
_output_shapes
:?????????221
/sequential_129/lstm_134/lstm_cell_134/Sigmoid_2?
,sequential_129/lstm_134/lstm_cell_134/Relu_1Relu/sequential_129/lstm_134/lstm_cell_134/add_1:z:0*
T0*'
_output_shapes
:?????????22.
,sequential_129/lstm_134/lstm_cell_134/Relu_1?
+sequential_129/lstm_134/lstm_cell_134/mul_2Mul3sequential_129/lstm_134/lstm_cell_134/Sigmoid_2:y:0:sequential_129/lstm_134/lstm_cell_134/Relu_1:activations:0*
T0*'
_output_shapes
:?????????22-
+sequential_129/lstm_134/lstm_cell_134/mul_2?
5sequential_129/lstm_134/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????2   27
5sequential_129/lstm_134/TensorArrayV2_1/element_shape?
'sequential_129/lstm_134/TensorArrayV2_1TensorListReserve>sequential_129/lstm_134/TensorArrayV2_1/element_shape:output:00sequential_129/lstm_134/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02)
'sequential_129/lstm_134/TensorArrayV2_1~
sequential_129/lstm_134/timeConst*
_output_shapes
: *
dtype0*
value	B : 2
sequential_129/lstm_134/time?
0sequential_129/lstm_134/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????22
0sequential_129/lstm_134/while/maximum_iterations?
*sequential_129/lstm_134/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2,
*sequential_129/lstm_134/while/loop_counter?
sequential_129/lstm_134/whileWhile3sequential_129/lstm_134/while/loop_counter:output:09sequential_129/lstm_134/while/maximum_iterations:output:0%sequential_129/lstm_134/time:output:00sequential_129/lstm_134/TensorArrayV2_1:handle:0&sequential_129/lstm_134/zeros:output:0(sequential_129/lstm_134/zeros_1:output:00sequential_129/lstm_134/strided_slice_1:output:0Osequential_129/lstm_134/TensorArrayUnstack/TensorListFromTensor:output_handle:0Dsequential_129_lstm_134_lstm_cell_134_matmul_readvariableop_resourceFsequential_129_lstm_134_lstm_cell_134_matmul_1_readvariableop_resourceEsequential_129_lstm_134_lstm_cell_134_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :?????????2:?????????2: : : : : *%
_read_only_resource_inputs
	
*6
body.R,
*sequential_129_lstm_134_while_body_1063495*6
cond.R,
*sequential_129_lstm_134_while_cond_1063494*K
output_shapes:
8: : : : :?????????2:?????????2: : : : : *
parallel_iterations 2
sequential_129/lstm_134/while?
Hsequential_129/lstm_134/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????2   2J
Hsequential_129/lstm_134/TensorArrayV2Stack/TensorListStack/element_shape?
:sequential_129/lstm_134/TensorArrayV2Stack/TensorListStackTensorListStack&sequential_129/lstm_134/while:output:3Qsequential_129/lstm_134/TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:?????????2*
element_dtype02<
:sequential_129/lstm_134/TensorArrayV2Stack/TensorListStack?
-sequential_129/lstm_134/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????2/
-sequential_129/lstm_134/strided_slice_3/stack?
/sequential_129/lstm_134/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 21
/sequential_129/lstm_134/strided_slice_3/stack_1?
/sequential_129/lstm_134/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:21
/sequential_129/lstm_134/strided_slice_3/stack_2?
'sequential_129/lstm_134/strided_slice_3StridedSliceCsequential_129/lstm_134/TensorArrayV2Stack/TensorListStack:tensor:06sequential_129/lstm_134/strided_slice_3/stack:output:08sequential_129/lstm_134/strided_slice_3/stack_1:output:08sequential_129/lstm_134/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????2*
shrink_axis_mask2)
'sequential_129/lstm_134/strided_slice_3?
(sequential_129/lstm_134/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2*
(sequential_129/lstm_134/transpose_1/perm?
#sequential_129/lstm_134/transpose_1	TransposeCsequential_129/lstm_134/TensorArrayV2Stack/TensorListStack:tensor:01sequential_129/lstm_134/transpose_1/perm:output:0*
T0*+
_output_shapes
:?????????22%
#sequential_129/lstm_134/transpose_1?
sequential_129/lstm_134/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2!
sequential_129/lstm_134/runtime?
.sequential_129/dense_128/MatMul/ReadVariableOpReadVariableOp7sequential_129_dense_128_matmul_readvariableop_resource*
_output_shapes

:2*
dtype020
.sequential_129/dense_128/MatMul/ReadVariableOp?
sequential_129/dense_128/MatMulMatMul0sequential_129/lstm_134/strided_slice_3:output:06sequential_129/dense_128/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2!
sequential_129/dense_128/MatMul?
/sequential_129/dense_128/BiasAdd/ReadVariableOpReadVariableOp8sequential_129_dense_128_biasadd_readvariableop_resource*
_output_shapes
:*
dtype021
/sequential_129/dense_128/BiasAdd/ReadVariableOp?
 sequential_129/dense_128/BiasAddBiasAdd)sequential_129/dense_128/MatMul:product:07sequential_129/dense_128/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2"
 sequential_129/dense_128/BiasAdd?
IdentityIdentity)sequential_129/dense_128/BiasAdd:output:00^sequential_129/dense_128/BiasAdd/ReadVariableOp/^sequential_129/dense_128/MatMul/ReadVariableOp=^sequential_129/lstm_134/lstm_cell_134/BiasAdd/ReadVariableOp<^sequential_129/lstm_134/lstm_cell_134/MatMul/ReadVariableOp>^sequential_129/lstm_134/lstm_cell_134/MatMul_1/ReadVariableOp^sequential_129/lstm_134/while*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:?????????:::::2b
/sequential_129/dense_128/BiasAdd/ReadVariableOp/sequential_129/dense_128/BiasAdd/ReadVariableOp2`
.sequential_129/dense_128/MatMul/ReadVariableOp.sequential_129/dense_128/MatMul/ReadVariableOp2|
<sequential_129/lstm_134/lstm_cell_134/BiasAdd/ReadVariableOp<sequential_129/lstm_134/lstm_cell_134/BiasAdd/ReadVariableOp2z
;sequential_129/lstm_134/lstm_cell_134/MatMul/ReadVariableOp;sequential_129/lstm_134/lstm_cell_134/MatMul/ReadVariableOp2~
=sequential_129/lstm_134/lstm_cell_134/MatMul_1/ReadVariableOp=sequential_129/lstm_134/lstm_cell_134/MatMul_1/ReadVariableOp2>
sequential_129/lstm_134/whilesequential_129/lstm_134/while:[ W
+
_output_shapes
:?????????
(
_user_specified_namelstm_134_input
?
?
K__inference_sequential_129_layer_call_and_return_conditional_losses_1064628

inputs
lstm_134_1064615
lstm_134_1064617
lstm_134_1064619
dense_128_1064622
dense_128_1064624
identity??!dense_128/StatefulPartitionedCall? lstm_134/StatefulPartitionedCall?
 lstm_134/StatefulPartitionedCallStatefulPartitionedCallinputslstm_134_1064615lstm_134_1064617lstm_134_1064619*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????2*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_lstm_134_layer_call_and_return_conditional_losses_10645052"
 lstm_134/StatefulPartitionedCall?
!dense_128/StatefulPartitionedCallStatefulPartitionedCall)lstm_134/StatefulPartitionedCall:output:0dense_128_1064622dense_128_1064624*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_dense_128_layer_call_and_return_conditional_losses_10645452#
!dense_128/StatefulPartitionedCall?
IdentityIdentity*dense_128/StatefulPartitionedCall:output:0"^dense_128/StatefulPartitionedCall!^lstm_134/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:?????????:::::2F
!dense_128/StatefulPartitionedCall!dense_128/StatefulPartitionedCall2D
 lstm_134/StatefulPartitionedCall lstm_134/StatefulPartitionedCall:S O
+
_output_shapes
:?????????
 
_user_specified_nameinputs
?C
?
while_body_1064420
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_08
4while_lstm_cell_134_matmul_readvariableop_resource_0:
6while_lstm_cell_134_matmul_1_readvariableop_resource_09
5while_lstm_cell_134_biasadd_readvariableop_resource_0
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor6
2while_lstm_cell_134_matmul_readvariableop_resource8
4while_lstm_cell_134_matmul_1_readvariableop_resource7
3while_lstm_cell_134_biasadd_readvariableop_resource??*while/lstm_cell_134/BiasAdd/ReadVariableOp?)while/lstm_cell_134/MatMul/ReadVariableOp?+while/lstm_cell_134/MatMul_1/ReadVariableOp?
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   29
7while/TensorArrayV2Read/TensorListGetItem/element_shape?
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:?????????*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItem?
)while/lstm_cell_134/MatMul/ReadVariableOpReadVariableOp4while_lstm_cell_134_matmul_readvariableop_resource_0*
_output_shapes
:	?*
dtype02+
)while/lstm_cell_134/MatMul/ReadVariableOp?
while/lstm_cell_134/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:01while/lstm_cell_134/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_134/MatMul?
+while/lstm_cell_134/MatMul_1/ReadVariableOpReadVariableOp6while_lstm_cell_134_matmul_1_readvariableop_resource_0*
_output_shapes
:	2?*
dtype02-
+while/lstm_cell_134/MatMul_1/ReadVariableOp?
while/lstm_cell_134/MatMul_1MatMulwhile_placeholder_23while/lstm_cell_134/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_134/MatMul_1?
while/lstm_cell_134/addAddV2$while/lstm_cell_134/MatMul:product:0&while/lstm_cell_134/MatMul_1:product:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_134/add?
*while/lstm_cell_134/BiasAdd/ReadVariableOpReadVariableOp5while_lstm_cell_134_biasadd_readvariableop_resource_0*
_output_shapes	
:?*
dtype02,
*while/lstm_cell_134/BiasAdd/ReadVariableOp?
while/lstm_cell_134/BiasAddBiasAddwhile/lstm_cell_134/add:z:02while/lstm_cell_134/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_134/BiasAddx
while/lstm_cell_134/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
while/lstm_cell_134/Const?
#while/lstm_cell_134/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2%
#while/lstm_cell_134/split/split_dim?
while/lstm_cell_134/splitSplit,while/lstm_cell_134/split/split_dim:output:0$while/lstm_cell_134/BiasAdd:output:0*
T0*`
_output_shapesN
L:?????????2:?????????2:?????????2:?????????2*
	num_split2
while/lstm_cell_134/split?
while/lstm_cell_134/SigmoidSigmoid"while/lstm_cell_134/split:output:0*
T0*'
_output_shapes
:?????????22
while/lstm_cell_134/Sigmoid?
while/lstm_cell_134/Sigmoid_1Sigmoid"while/lstm_cell_134/split:output:1*
T0*'
_output_shapes
:?????????22
while/lstm_cell_134/Sigmoid_1?
while/lstm_cell_134/mulMul!while/lstm_cell_134/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:?????????22
while/lstm_cell_134/mul?
while/lstm_cell_134/ReluRelu"while/lstm_cell_134/split:output:2*
T0*'
_output_shapes
:?????????22
while/lstm_cell_134/Relu?
while/lstm_cell_134/mul_1Mulwhile/lstm_cell_134/Sigmoid:y:0&while/lstm_cell_134/Relu:activations:0*
T0*'
_output_shapes
:?????????22
while/lstm_cell_134/mul_1?
while/lstm_cell_134/add_1AddV2while/lstm_cell_134/mul:z:0while/lstm_cell_134/mul_1:z:0*
T0*'
_output_shapes
:?????????22
while/lstm_cell_134/add_1?
while/lstm_cell_134/Sigmoid_2Sigmoid"while/lstm_cell_134/split:output:3*
T0*'
_output_shapes
:?????????22
while/lstm_cell_134/Sigmoid_2?
while/lstm_cell_134/Relu_1Reluwhile/lstm_cell_134/add_1:z:0*
T0*'
_output_shapes
:?????????22
while/lstm_cell_134/Relu_1?
while/lstm_cell_134/mul_2Mul!while/lstm_cell_134/Sigmoid_2:y:0(while/lstm_cell_134/Relu_1:activations:0*
T0*'
_output_shapes
:?????????22
while/lstm_cell_134/mul_2?
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_134/mul_2:z:0*
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
while/add_1?
while/IdentityIdentitywhile/add_1:z:0+^while/lstm_cell_134/BiasAdd/ReadVariableOp*^while/lstm_cell_134/MatMul/ReadVariableOp,^while/lstm_cell_134/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity?
while/Identity_1Identitywhile_while_maximum_iterations+^while/lstm_cell_134/BiasAdd/ReadVariableOp*^while/lstm_cell_134/MatMul/ReadVariableOp,^while/lstm_cell_134/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_1?
while/Identity_2Identitywhile/add:z:0+^while/lstm_cell_134/BiasAdd/ReadVariableOp*^while/lstm_cell_134/MatMul/ReadVariableOp,^while/lstm_cell_134/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_2?
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0+^while/lstm_cell_134/BiasAdd/ReadVariableOp*^while/lstm_cell_134/MatMul/ReadVariableOp,^while/lstm_cell_134/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_3?
while/Identity_4Identitywhile/lstm_cell_134/mul_2:z:0+^while/lstm_cell_134/BiasAdd/ReadVariableOp*^while/lstm_cell_134/MatMul/ReadVariableOp,^while/lstm_cell_134/MatMul_1/ReadVariableOp*
T0*'
_output_shapes
:?????????22
while/Identity_4?
while/Identity_5Identitywhile/lstm_cell_134/add_1:z:0+^while/lstm_cell_134/BiasAdd/ReadVariableOp*^while/lstm_cell_134/MatMul/ReadVariableOp,^while/lstm_cell_134/MatMul_1/ReadVariableOp*
T0*'
_output_shapes
:?????????22
while/Identity_5")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"l
3while_lstm_cell_134_biasadd_readvariableop_resource5while_lstm_cell_134_biasadd_readvariableop_resource_0"n
4while_lstm_cell_134_matmul_1_readvariableop_resource6while_lstm_cell_134_matmul_1_readvariableop_resource_0"j
2while_lstm_cell_134_matmul_readvariableop_resource4while_lstm_cell_134_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"?
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*Q
_input_shapes@
>: : : : :?????????2:?????????2: : :::2X
*while/lstm_cell_134/BiasAdd/ReadVariableOp*while/lstm_cell_134/BiasAdd/ReadVariableOp2V
)while/lstm_cell_134/MatMul/ReadVariableOp)while/lstm_cell_134/MatMul/ReadVariableOp2Z
+while/lstm_cell_134/MatMul_1/ReadVariableOp+while/lstm_cell_134/MatMul_1/ReadVariableOp: 
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
:?????????2:-)
'
_output_shapes
:?????????2:

_output_shapes
: :

_output_shapes
: 
?
?
J__inference_lstm_cell_134_layer_call_and_return_conditional_losses_1065755

inputs
states_0
states_1"
matmul_readvariableop_resource$
 matmul_1_readvariableop_resource#
biasadd_readvariableop_resource
identity

identity_1

identity_2??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?MatMul_1/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	?*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
MatMul?
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource*
_output_shapes
:	2?*
dtype02
MatMul_1/ReadVariableOp|
MatMul_1MatMulstates_0MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2

MatMul_1l
addAddV2MatMul:product:0MatMul_1:product:0*
T0*(
_output_shapes
:??????????2
add?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOpy
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2	
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
split/split_dim?
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*`
_output_shapesN
L:?????????2:?????????2:?????????2:?????????2*
	num_split2
split_
SigmoidSigmoidsplit:output:0*
T0*'
_output_shapes
:?????????22	
Sigmoidc
	Sigmoid_1Sigmoidsplit:output:1*
T0*'
_output_shapes
:?????????22
	Sigmoid_1\
mulMulSigmoid_1:y:0states_1*
T0*'
_output_shapes
:?????????22
mulV
ReluRelusplit:output:2*
T0*'
_output_shapes
:?????????22
Reluh
mul_1MulSigmoid:y:0Relu:activations:0*
T0*'
_output_shapes
:?????????22
mul_1]
add_1AddV2mul:z:0	mul_1:z:0*
T0*'
_output_shapes
:?????????22
add_1c
	Sigmoid_2Sigmoidsplit:output:3*
T0*'
_output_shapes
:?????????22
	Sigmoid_2U
Relu_1Relu	add_1:z:0*
T0*'
_output_shapes
:?????????22
Relu_1l
mul_2MulSigmoid_2:y:0Relu_1:activations:0*
T0*'
_output_shapes
:?????????22
mul_2?
IdentityIdentity	mul_2:z:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*'
_output_shapes
:?????????22

Identity?

Identity_1Identity	mul_2:z:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*'
_output_shapes
:?????????22

Identity_1?

Identity_2Identity	add_1:z:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*'
_output_shapes
:?????????22

Identity_2"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*X
_input_shapesG
E:?????????:?????????2:?????????2:::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs:QM
'
_output_shapes
:?????????2
"
_user_specified_name
states/0:QM
'
_output_shapes
:?????????2
"
_user_specified_name
states/1
?

?
lstm_134_while_cond_1064733.
*lstm_134_while_lstm_134_while_loop_counter4
0lstm_134_while_lstm_134_while_maximum_iterations
lstm_134_while_placeholder 
lstm_134_while_placeholder_1 
lstm_134_while_placeholder_2 
lstm_134_while_placeholder_30
,lstm_134_while_less_lstm_134_strided_slice_1G
Clstm_134_while_lstm_134_while_cond_1064733___redundant_placeholder0G
Clstm_134_while_lstm_134_while_cond_1064733___redundant_placeholder1G
Clstm_134_while_lstm_134_while_cond_1064733___redundant_placeholder2G
Clstm_134_while_lstm_134_while_cond_1064733___redundant_placeholder3
lstm_134_while_identity
?
lstm_134/while/LessLesslstm_134_while_placeholder,lstm_134_while_less_lstm_134_strided_slice_1*
T0*
_output_shapes
: 2
lstm_134/while/Lessx
lstm_134/while/IdentityIdentitylstm_134/while/Less:z:0*
T0
*
_output_shapes
: 2
lstm_134/while/Identity";
lstm_134_while_identity lstm_134/while/Identity:output:0*S
_input_shapesB
@: : : : :?????????2:?????????2: ::::: 
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
:?????????2:-)
'
_output_shapes
:?????????2:

_output_shapes
: :

_output_shapes
:
?[
?
E__inference_lstm_134_layer_call_and_return_conditional_losses_1065495

inputs0
,lstm_cell_134_matmul_readvariableop_resource2
.lstm_cell_134_matmul_1_readvariableop_resource1
-lstm_cell_134_biasadd_readvariableop_resource
identity??$lstm_cell_134/BiasAdd/ReadVariableOp?#lstm_cell_134/MatMul/ReadVariableOp?%lstm_cell_134/MatMul_1/ReadVariableOp?whileD
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
strided_slice/stack_2?
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
value	B :22
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
B :?2
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
value	B :22
zeros/packed/1?
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
:?????????22
zeros`
zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B :22
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
B :?2
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
value	B :22
zeros_1/packed/1?
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
:?????????22	
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
:?????????2
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
strided_slice_1/stack_2?
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1?
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
TensorArrayV2/element_shape?
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2?
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   27
5TensorArrayUnstack/TensorListFromTensor/element_shape?
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
strided_slice_2/stack_2?
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*
shrink_axis_mask2
strided_slice_2?
#lstm_cell_134/MatMul/ReadVariableOpReadVariableOp,lstm_cell_134_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype02%
#lstm_cell_134/MatMul/ReadVariableOp?
lstm_cell_134/MatMulMatMulstrided_slice_2:output:0+lstm_cell_134/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
lstm_cell_134/MatMul?
%lstm_cell_134/MatMul_1/ReadVariableOpReadVariableOp.lstm_cell_134_matmul_1_readvariableop_resource*
_output_shapes
:	2?*
dtype02'
%lstm_cell_134/MatMul_1/ReadVariableOp?
lstm_cell_134/MatMul_1MatMulzeros:output:0-lstm_cell_134/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
lstm_cell_134/MatMul_1?
lstm_cell_134/addAddV2lstm_cell_134/MatMul:product:0 lstm_cell_134/MatMul_1:product:0*
T0*(
_output_shapes
:??????????2
lstm_cell_134/add?
$lstm_cell_134/BiasAdd/ReadVariableOpReadVariableOp-lstm_cell_134_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02&
$lstm_cell_134/BiasAdd/ReadVariableOp?
lstm_cell_134/BiasAddBiasAddlstm_cell_134/add:z:0,lstm_cell_134/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
lstm_cell_134/BiasAddl
lstm_cell_134/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_134/Const?
lstm_cell_134/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_134/split/split_dim?
lstm_cell_134/splitSplit&lstm_cell_134/split/split_dim:output:0lstm_cell_134/BiasAdd:output:0*
T0*`
_output_shapesN
L:?????????2:?????????2:?????????2:?????????2*
	num_split2
lstm_cell_134/split?
lstm_cell_134/SigmoidSigmoidlstm_cell_134/split:output:0*
T0*'
_output_shapes
:?????????22
lstm_cell_134/Sigmoid?
lstm_cell_134/Sigmoid_1Sigmoidlstm_cell_134/split:output:1*
T0*'
_output_shapes
:?????????22
lstm_cell_134/Sigmoid_1?
lstm_cell_134/mulMullstm_cell_134/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:?????????22
lstm_cell_134/mul?
lstm_cell_134/ReluRelulstm_cell_134/split:output:2*
T0*'
_output_shapes
:?????????22
lstm_cell_134/Relu?
lstm_cell_134/mul_1Mullstm_cell_134/Sigmoid:y:0 lstm_cell_134/Relu:activations:0*
T0*'
_output_shapes
:?????????22
lstm_cell_134/mul_1?
lstm_cell_134/add_1AddV2lstm_cell_134/mul:z:0lstm_cell_134/mul_1:z:0*
T0*'
_output_shapes
:?????????22
lstm_cell_134/add_1?
lstm_cell_134/Sigmoid_2Sigmoidlstm_cell_134/split:output:3*
T0*'
_output_shapes
:?????????22
lstm_cell_134/Sigmoid_2
lstm_cell_134/Relu_1Relulstm_cell_134/add_1:z:0*
T0*'
_output_shapes
:?????????22
lstm_cell_134/Relu_1?
lstm_cell_134/mul_2Mullstm_cell_134/Sigmoid_2:y:0"lstm_cell_134/Relu_1:activations:0*
T0*'
_output_shapes
:?????????22
lstm_cell_134/mul_2?
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????2   2
TensorArrayV2_1/element_shape?
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
?????????2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counter?
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0,lstm_cell_134_matmul_readvariableop_resource.lstm_cell_134_matmul_1_readvariableop_resource-lstm_cell_134_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :?????????2:?????????2: : : : : *%
_read_only_resource_inputs
	
*
bodyR
while_body_1065410*
condR
while_cond_1065409*K
output_shapes:
8: : : : :?????????2:?????????2: : : : : *
parallel_iterations 2
while?
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????2   22
0TensorArrayV2Stack/TensorListStack/element_shape?
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:?????????2*
element_dtype02$
"TensorArrayV2Stack/TensorListStack?
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????2
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
strided_slice_3/stack_2?
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????2*
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/perm?
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*+
_output_shapes
:?????????22
transpose_1f
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2	
runtime?
IdentityIdentitystrided_slice_3:output:0%^lstm_cell_134/BiasAdd/ReadVariableOp$^lstm_cell_134/MatMul/ReadVariableOp&^lstm_cell_134/MatMul_1/ReadVariableOp^while*
T0*'
_output_shapes
:?????????22

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????:::2L
$lstm_cell_134/BiasAdd/ReadVariableOp$lstm_cell_134/BiasAdd/ReadVariableOp2J
#lstm_cell_134/MatMul/ReadVariableOp#lstm_cell_134/MatMul/ReadVariableOp2N
%lstm_cell_134/MatMul_1/ReadVariableOp%lstm_cell_134/MatMul_1/ReadVariableOp2
whilewhile:S O
+
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
J__inference_lstm_cell_134_layer_call_and_return_conditional_losses_1063659

inputs

states
states_1"
matmul_readvariableop_resource$
 matmul_1_readvariableop_resource#
biasadd_readvariableop_resource
identity

identity_1

identity_2??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?MatMul_1/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	?*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
MatMul?
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource*
_output_shapes
:	2?*
dtype02
MatMul_1/ReadVariableOpz
MatMul_1MatMulstatesMatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2

MatMul_1l
addAddV2MatMul:product:0MatMul_1:product:0*
T0*(
_output_shapes
:??????????2
add?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOpy
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2	
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
split/split_dim?
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*`
_output_shapesN
L:?????????2:?????????2:?????????2:?????????2*
	num_split2
split_
SigmoidSigmoidsplit:output:0*
T0*'
_output_shapes
:?????????22	
Sigmoidc
	Sigmoid_1Sigmoidsplit:output:1*
T0*'
_output_shapes
:?????????22
	Sigmoid_1\
mulMulSigmoid_1:y:0states_1*
T0*'
_output_shapes
:?????????22
mulV
ReluRelusplit:output:2*
T0*'
_output_shapes
:?????????22
Reluh
mul_1MulSigmoid:y:0Relu:activations:0*
T0*'
_output_shapes
:?????????22
mul_1]
add_1AddV2mul:z:0	mul_1:z:0*
T0*'
_output_shapes
:?????????22
add_1c
	Sigmoid_2Sigmoidsplit:output:3*
T0*'
_output_shapes
:?????????22
	Sigmoid_2U
Relu_1Relu	add_1:z:0*
T0*'
_output_shapes
:?????????22
Relu_1l
mul_2MulSigmoid_2:y:0Relu_1:activations:0*
T0*'
_output_shapes
:?????????22
mul_2?
IdentityIdentity	mul_2:z:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*'
_output_shapes
:?????????22

Identity?

Identity_1Identity	mul_2:z:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*'
_output_shapes
:?????????22

Identity_1?

Identity_2Identity	add_1:z:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*'
_output_shapes
:?????????22

Identity_2"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*X
_input_shapesG
E:?????????:?????????2:?????????2:::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs:OK
'
_output_shapes
:?????????2
 
_user_specified_namestates:OK
'
_output_shapes
:?????????2
 
_user_specified_namestates
?
?
/__inference_lstm_cell_134_layer_call_fn_1065789

inputs
states_0
states_1
unknown
	unknown_0
	unknown_1
identity

identity_1

identity_2??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsstates_0states_1unknown	unknown_0	unknown_1*
Tin

2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:?????????2:?????????2:?????????2*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *S
fNRL
J__inference_lstm_cell_134_layer_call_and_return_conditional_losses_10636922
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????22

Identity?

Identity_1Identity StatefulPartitionedCall:output:1^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????22

Identity_1?

Identity_2Identity StatefulPartitionedCall:output:2^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????22

Identity_2"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*X
_input_shapesG
E:?????????:?????????2:?????????2:::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs:QM
'
_output_shapes
:?????????2
"
_user_specified_name
states/0:QM
'
_output_shapes
:?????????2
"
_user_specified_name
states/1
?
?
+__inference_dense_128_layer_call_fn_1065689

inputs
unknown
	unknown_0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_dense_128_layer_call_and_return_conditional_losses_10645452
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????2::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????2
 
_user_specified_nameinputs
?7
?

 __inference__traced_save_1065878
file_prefix/
+savev2_dense_128_kernel_read_readvariableop-
)savev2_dense_128_bias_read_readvariableop(
$savev2_adam_iter_read_readvariableop	*
&savev2_adam_beta_1_read_readvariableop*
&savev2_adam_beta_2_read_readvariableop)
%savev2_adam_decay_read_readvariableop1
-savev2_adam_learning_rate_read_readvariableop<
8savev2_lstm_134_lstm_cell_134_kernel_read_readvariableopF
Bsavev2_lstm_134_lstm_cell_134_recurrent_kernel_read_readvariableop:
6savev2_lstm_134_lstm_cell_134_bias_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop6
2savev2_adam_dense_128_kernel_m_read_readvariableop4
0savev2_adam_dense_128_bias_m_read_readvariableopC
?savev2_adam_lstm_134_lstm_cell_134_kernel_m_read_readvariableopM
Isavev2_adam_lstm_134_lstm_cell_134_recurrent_kernel_m_read_readvariableopA
=savev2_adam_lstm_134_lstm_cell_134_bias_m_read_readvariableop6
2savev2_adam_dense_128_kernel_v_read_readvariableop4
0savev2_adam_dense_128_bias_v_read_readvariableopC
?savev2_adam_lstm_134_lstm_cell_134_kernel_v_read_readvariableopM
Isavev2_adam_lstm_134_lstm_cell_134_recurrent_kernel_v_read_readvariableopA
=savev2_adam_lstm_134_lstm_cell_134_bias_v_read_readvariableop
savev2_const

identity_1??MergeV2Checkpoints?
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
Const_1?
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
ShardedFilename/shard?
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilename?
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*?

value?
B?
B6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_names?
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*A
value8B6B B B B B B B B B B B B B B B B B B B B B B B 2
SaveV2/shape_and_slices?

SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0+savev2_dense_128_kernel_read_readvariableop)savev2_dense_128_bias_read_readvariableop$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop-savev2_adam_learning_rate_read_readvariableop8savev2_lstm_134_lstm_cell_134_kernel_read_readvariableopBsavev2_lstm_134_lstm_cell_134_recurrent_kernel_read_readvariableop6savev2_lstm_134_lstm_cell_134_bias_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop2savev2_adam_dense_128_kernel_m_read_readvariableop0savev2_adam_dense_128_bias_m_read_readvariableop?savev2_adam_lstm_134_lstm_cell_134_kernel_m_read_readvariableopIsavev2_adam_lstm_134_lstm_cell_134_recurrent_kernel_m_read_readvariableop=savev2_adam_lstm_134_lstm_cell_134_bias_m_read_readvariableop2savev2_adam_dense_128_kernel_v_read_readvariableop0savev2_adam_dense_128_bias_v_read_readvariableop?savev2_adam_lstm_134_lstm_cell_134_kernel_v_read_readvariableopIsavev2_adam_lstm_134_lstm_cell_134_recurrent_kernel_v_read_readvariableop=savev2_adam_lstm_134_lstm_cell_134_bias_v_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *%
dtypes
2	2
SaveV2?
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:2(
&MergeV2Checkpoints/checkpoint_prefixes?
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

identity_1Identity_1:output:0*?
_input_shapes?
?: :2:: : : : : :	?:	2?:?: : :2::	?:	2?:?:2::	?:	2?:?: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:$ 

_output_shapes

:2: 
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
:	?:%	!

_output_shapes
:	2?:!


_output_shapes	
:?:

_output_shapes
: :

_output_shapes
: :$ 

_output_shapes

:2: 

_output_shapes
::%!

_output_shapes
:	?:%!

_output_shapes
:	2?:!

_output_shapes	
:?:$ 

_output_shapes

:2: 

_output_shapes
::%!

_output_shapes
:	?:%!

_output_shapes
:	2?:!

_output_shapes	
:?:

_output_shapes
: 
?g
?
*sequential_129_lstm_134_while_body_1063495L
Hsequential_129_lstm_134_while_sequential_129_lstm_134_while_loop_counterR
Nsequential_129_lstm_134_while_sequential_129_lstm_134_while_maximum_iterations-
)sequential_129_lstm_134_while_placeholder/
+sequential_129_lstm_134_while_placeholder_1/
+sequential_129_lstm_134_while_placeholder_2/
+sequential_129_lstm_134_while_placeholder_3K
Gsequential_129_lstm_134_while_sequential_129_lstm_134_strided_slice_1_0?
?sequential_129_lstm_134_while_tensorarrayv2read_tensorlistgetitem_sequential_129_lstm_134_tensorarrayunstack_tensorlistfromtensor_0P
Lsequential_129_lstm_134_while_lstm_cell_134_matmul_readvariableop_resource_0R
Nsequential_129_lstm_134_while_lstm_cell_134_matmul_1_readvariableop_resource_0Q
Msequential_129_lstm_134_while_lstm_cell_134_biasadd_readvariableop_resource_0*
&sequential_129_lstm_134_while_identity,
(sequential_129_lstm_134_while_identity_1,
(sequential_129_lstm_134_while_identity_2,
(sequential_129_lstm_134_while_identity_3,
(sequential_129_lstm_134_while_identity_4,
(sequential_129_lstm_134_while_identity_5I
Esequential_129_lstm_134_while_sequential_129_lstm_134_strided_slice_1?
?sequential_129_lstm_134_while_tensorarrayv2read_tensorlistgetitem_sequential_129_lstm_134_tensorarrayunstack_tensorlistfromtensorN
Jsequential_129_lstm_134_while_lstm_cell_134_matmul_readvariableop_resourceP
Lsequential_129_lstm_134_while_lstm_cell_134_matmul_1_readvariableop_resourceO
Ksequential_129_lstm_134_while_lstm_cell_134_biasadd_readvariableop_resource??Bsequential_129/lstm_134/while/lstm_cell_134/BiasAdd/ReadVariableOp?Asequential_129/lstm_134/while/lstm_cell_134/MatMul/ReadVariableOp?Csequential_129/lstm_134/while/lstm_cell_134/MatMul_1/ReadVariableOp?
Osequential_129/lstm_134/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   2Q
Osequential_129/lstm_134/while/TensorArrayV2Read/TensorListGetItem/element_shape?
Asequential_129/lstm_134/while/TensorArrayV2Read/TensorListGetItemTensorListGetItem?sequential_129_lstm_134_while_tensorarrayv2read_tensorlistgetitem_sequential_129_lstm_134_tensorarrayunstack_tensorlistfromtensor_0)sequential_129_lstm_134_while_placeholderXsequential_129/lstm_134/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:?????????*
element_dtype02C
Asequential_129/lstm_134/while/TensorArrayV2Read/TensorListGetItem?
Asequential_129/lstm_134/while/lstm_cell_134/MatMul/ReadVariableOpReadVariableOpLsequential_129_lstm_134_while_lstm_cell_134_matmul_readvariableop_resource_0*
_output_shapes
:	?*
dtype02C
Asequential_129/lstm_134/while/lstm_cell_134/MatMul/ReadVariableOp?
2sequential_129/lstm_134/while/lstm_cell_134/MatMulMatMulHsequential_129/lstm_134/while/TensorArrayV2Read/TensorListGetItem:item:0Isequential_129/lstm_134/while/lstm_cell_134/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????24
2sequential_129/lstm_134/while/lstm_cell_134/MatMul?
Csequential_129/lstm_134/while/lstm_cell_134/MatMul_1/ReadVariableOpReadVariableOpNsequential_129_lstm_134_while_lstm_cell_134_matmul_1_readvariableop_resource_0*
_output_shapes
:	2?*
dtype02E
Csequential_129/lstm_134/while/lstm_cell_134/MatMul_1/ReadVariableOp?
4sequential_129/lstm_134/while/lstm_cell_134/MatMul_1MatMul+sequential_129_lstm_134_while_placeholder_2Ksequential_129/lstm_134/while/lstm_cell_134/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????26
4sequential_129/lstm_134/while/lstm_cell_134/MatMul_1?
/sequential_129/lstm_134/while/lstm_cell_134/addAddV2<sequential_129/lstm_134/while/lstm_cell_134/MatMul:product:0>sequential_129/lstm_134/while/lstm_cell_134/MatMul_1:product:0*
T0*(
_output_shapes
:??????????21
/sequential_129/lstm_134/while/lstm_cell_134/add?
Bsequential_129/lstm_134/while/lstm_cell_134/BiasAdd/ReadVariableOpReadVariableOpMsequential_129_lstm_134_while_lstm_cell_134_biasadd_readvariableop_resource_0*
_output_shapes	
:?*
dtype02D
Bsequential_129/lstm_134/while/lstm_cell_134/BiasAdd/ReadVariableOp?
3sequential_129/lstm_134/while/lstm_cell_134/BiasAddBiasAdd3sequential_129/lstm_134/while/lstm_cell_134/add:z:0Jsequential_129/lstm_134/while/lstm_cell_134/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????25
3sequential_129/lstm_134/while/lstm_cell_134/BiasAdd?
1sequential_129/lstm_134/while/lstm_cell_134/ConstConst*
_output_shapes
: *
dtype0*
value	B :23
1sequential_129/lstm_134/while/lstm_cell_134/Const?
;sequential_129/lstm_134/while/lstm_cell_134/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2=
;sequential_129/lstm_134/while/lstm_cell_134/split/split_dim?
1sequential_129/lstm_134/while/lstm_cell_134/splitSplitDsequential_129/lstm_134/while/lstm_cell_134/split/split_dim:output:0<sequential_129/lstm_134/while/lstm_cell_134/BiasAdd:output:0*
T0*`
_output_shapesN
L:?????????2:?????????2:?????????2:?????????2*
	num_split23
1sequential_129/lstm_134/while/lstm_cell_134/split?
3sequential_129/lstm_134/while/lstm_cell_134/SigmoidSigmoid:sequential_129/lstm_134/while/lstm_cell_134/split:output:0*
T0*'
_output_shapes
:?????????225
3sequential_129/lstm_134/while/lstm_cell_134/Sigmoid?
5sequential_129/lstm_134/while/lstm_cell_134/Sigmoid_1Sigmoid:sequential_129/lstm_134/while/lstm_cell_134/split:output:1*
T0*'
_output_shapes
:?????????227
5sequential_129/lstm_134/while/lstm_cell_134/Sigmoid_1?
/sequential_129/lstm_134/while/lstm_cell_134/mulMul9sequential_129/lstm_134/while/lstm_cell_134/Sigmoid_1:y:0+sequential_129_lstm_134_while_placeholder_3*
T0*'
_output_shapes
:?????????221
/sequential_129/lstm_134/while/lstm_cell_134/mul?
0sequential_129/lstm_134/while/lstm_cell_134/ReluRelu:sequential_129/lstm_134/while/lstm_cell_134/split:output:2*
T0*'
_output_shapes
:?????????222
0sequential_129/lstm_134/while/lstm_cell_134/Relu?
1sequential_129/lstm_134/while/lstm_cell_134/mul_1Mul7sequential_129/lstm_134/while/lstm_cell_134/Sigmoid:y:0>sequential_129/lstm_134/while/lstm_cell_134/Relu:activations:0*
T0*'
_output_shapes
:?????????223
1sequential_129/lstm_134/while/lstm_cell_134/mul_1?
1sequential_129/lstm_134/while/lstm_cell_134/add_1AddV23sequential_129/lstm_134/while/lstm_cell_134/mul:z:05sequential_129/lstm_134/while/lstm_cell_134/mul_1:z:0*
T0*'
_output_shapes
:?????????223
1sequential_129/lstm_134/while/lstm_cell_134/add_1?
5sequential_129/lstm_134/while/lstm_cell_134/Sigmoid_2Sigmoid:sequential_129/lstm_134/while/lstm_cell_134/split:output:3*
T0*'
_output_shapes
:?????????227
5sequential_129/lstm_134/while/lstm_cell_134/Sigmoid_2?
2sequential_129/lstm_134/while/lstm_cell_134/Relu_1Relu5sequential_129/lstm_134/while/lstm_cell_134/add_1:z:0*
T0*'
_output_shapes
:?????????224
2sequential_129/lstm_134/while/lstm_cell_134/Relu_1?
1sequential_129/lstm_134/while/lstm_cell_134/mul_2Mul9sequential_129/lstm_134/while/lstm_cell_134/Sigmoid_2:y:0@sequential_129/lstm_134/while/lstm_cell_134/Relu_1:activations:0*
T0*'
_output_shapes
:?????????223
1sequential_129/lstm_134/while/lstm_cell_134/mul_2?
Bsequential_129/lstm_134/while/TensorArrayV2Write/TensorListSetItemTensorListSetItem+sequential_129_lstm_134_while_placeholder_1)sequential_129_lstm_134_while_placeholder5sequential_129/lstm_134/while/lstm_cell_134/mul_2:z:0*
_output_shapes
: *
element_dtype02D
Bsequential_129/lstm_134/while/TensorArrayV2Write/TensorListSetItem?
#sequential_129/lstm_134/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2%
#sequential_129/lstm_134/while/add/y?
!sequential_129/lstm_134/while/addAddV2)sequential_129_lstm_134_while_placeholder,sequential_129/lstm_134/while/add/y:output:0*
T0*
_output_shapes
: 2#
!sequential_129/lstm_134/while/add?
%sequential_129/lstm_134/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2'
%sequential_129/lstm_134/while/add_1/y?
#sequential_129/lstm_134/while/add_1AddV2Hsequential_129_lstm_134_while_sequential_129_lstm_134_while_loop_counter.sequential_129/lstm_134/while/add_1/y:output:0*
T0*
_output_shapes
: 2%
#sequential_129/lstm_134/while/add_1?
&sequential_129/lstm_134/while/IdentityIdentity'sequential_129/lstm_134/while/add_1:z:0C^sequential_129/lstm_134/while/lstm_cell_134/BiasAdd/ReadVariableOpB^sequential_129/lstm_134/while/lstm_cell_134/MatMul/ReadVariableOpD^sequential_129/lstm_134/while/lstm_cell_134/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2(
&sequential_129/lstm_134/while/Identity?
(sequential_129/lstm_134/while/Identity_1IdentityNsequential_129_lstm_134_while_sequential_129_lstm_134_while_maximum_iterationsC^sequential_129/lstm_134/while/lstm_cell_134/BiasAdd/ReadVariableOpB^sequential_129/lstm_134/while/lstm_cell_134/MatMul/ReadVariableOpD^sequential_129/lstm_134/while/lstm_cell_134/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2*
(sequential_129/lstm_134/while/Identity_1?
(sequential_129/lstm_134/while/Identity_2Identity%sequential_129/lstm_134/while/add:z:0C^sequential_129/lstm_134/while/lstm_cell_134/BiasAdd/ReadVariableOpB^sequential_129/lstm_134/while/lstm_cell_134/MatMul/ReadVariableOpD^sequential_129/lstm_134/while/lstm_cell_134/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2*
(sequential_129/lstm_134/while/Identity_2?
(sequential_129/lstm_134/while/Identity_3IdentityRsequential_129/lstm_134/while/TensorArrayV2Write/TensorListSetItem:output_handle:0C^sequential_129/lstm_134/while/lstm_cell_134/BiasAdd/ReadVariableOpB^sequential_129/lstm_134/while/lstm_cell_134/MatMul/ReadVariableOpD^sequential_129/lstm_134/while/lstm_cell_134/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2*
(sequential_129/lstm_134/while/Identity_3?
(sequential_129/lstm_134/while/Identity_4Identity5sequential_129/lstm_134/while/lstm_cell_134/mul_2:z:0C^sequential_129/lstm_134/while/lstm_cell_134/BiasAdd/ReadVariableOpB^sequential_129/lstm_134/while/lstm_cell_134/MatMul/ReadVariableOpD^sequential_129/lstm_134/while/lstm_cell_134/MatMul_1/ReadVariableOp*
T0*'
_output_shapes
:?????????22*
(sequential_129/lstm_134/while/Identity_4?
(sequential_129/lstm_134/while/Identity_5Identity5sequential_129/lstm_134/while/lstm_cell_134/add_1:z:0C^sequential_129/lstm_134/while/lstm_cell_134/BiasAdd/ReadVariableOpB^sequential_129/lstm_134/while/lstm_cell_134/MatMul/ReadVariableOpD^sequential_129/lstm_134/while/lstm_cell_134/MatMul_1/ReadVariableOp*
T0*'
_output_shapes
:?????????22*
(sequential_129/lstm_134/while/Identity_5"Y
&sequential_129_lstm_134_while_identity/sequential_129/lstm_134/while/Identity:output:0"]
(sequential_129_lstm_134_while_identity_11sequential_129/lstm_134/while/Identity_1:output:0"]
(sequential_129_lstm_134_while_identity_21sequential_129/lstm_134/while/Identity_2:output:0"]
(sequential_129_lstm_134_while_identity_31sequential_129/lstm_134/while/Identity_3:output:0"]
(sequential_129_lstm_134_while_identity_41sequential_129/lstm_134/while/Identity_4:output:0"]
(sequential_129_lstm_134_while_identity_51sequential_129/lstm_134/while/Identity_5:output:0"?
Ksequential_129_lstm_134_while_lstm_cell_134_biasadd_readvariableop_resourceMsequential_129_lstm_134_while_lstm_cell_134_biasadd_readvariableop_resource_0"?
Lsequential_129_lstm_134_while_lstm_cell_134_matmul_1_readvariableop_resourceNsequential_129_lstm_134_while_lstm_cell_134_matmul_1_readvariableop_resource_0"?
Jsequential_129_lstm_134_while_lstm_cell_134_matmul_readvariableop_resourceLsequential_129_lstm_134_while_lstm_cell_134_matmul_readvariableop_resource_0"?
Esequential_129_lstm_134_while_sequential_129_lstm_134_strided_slice_1Gsequential_129_lstm_134_while_sequential_129_lstm_134_strided_slice_1_0"?
?sequential_129_lstm_134_while_tensorarrayv2read_tensorlistgetitem_sequential_129_lstm_134_tensorarrayunstack_tensorlistfromtensor?sequential_129_lstm_134_while_tensorarrayv2read_tensorlistgetitem_sequential_129_lstm_134_tensorarrayunstack_tensorlistfromtensor_0*Q
_input_shapes@
>: : : : :?????????2:?????????2: : :::2?
Bsequential_129/lstm_134/while/lstm_cell_134/BiasAdd/ReadVariableOpBsequential_129/lstm_134/while/lstm_cell_134/BiasAdd/ReadVariableOp2?
Asequential_129/lstm_134/while/lstm_cell_134/MatMul/ReadVariableOpAsequential_129/lstm_134/while/lstm_cell_134/MatMul/ReadVariableOp2?
Csequential_129/lstm_134/while/lstm_cell_134/MatMul_1/ReadVariableOpCsequential_129/lstm_134/while/lstm_cell_134/MatMul_1/ReadVariableOp: 
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
:?????????2:-)
'
_output_shapes
:?????????2:

_output_shapes
: :

_output_shapes
: 
?
?
*sequential_129_lstm_134_while_cond_1063494L
Hsequential_129_lstm_134_while_sequential_129_lstm_134_while_loop_counterR
Nsequential_129_lstm_134_while_sequential_129_lstm_134_while_maximum_iterations-
)sequential_129_lstm_134_while_placeholder/
+sequential_129_lstm_134_while_placeholder_1/
+sequential_129_lstm_134_while_placeholder_2/
+sequential_129_lstm_134_while_placeholder_3N
Jsequential_129_lstm_134_while_less_sequential_129_lstm_134_strided_slice_1e
asequential_129_lstm_134_while_sequential_129_lstm_134_while_cond_1063494___redundant_placeholder0e
asequential_129_lstm_134_while_sequential_129_lstm_134_while_cond_1063494___redundant_placeholder1e
asequential_129_lstm_134_while_sequential_129_lstm_134_while_cond_1063494___redundant_placeholder2e
asequential_129_lstm_134_while_sequential_129_lstm_134_while_cond_1063494___redundant_placeholder3*
&sequential_129_lstm_134_while_identity
?
"sequential_129/lstm_134/while/LessLess)sequential_129_lstm_134_while_placeholderJsequential_129_lstm_134_while_less_sequential_129_lstm_134_strided_slice_1*
T0*
_output_shapes
: 2$
"sequential_129/lstm_134/while/Less?
&sequential_129/lstm_134/while/IdentityIdentity&sequential_129/lstm_134/while/Less:z:0*
T0
*
_output_shapes
: 2(
&sequential_129/lstm_134/while/Identity"Y
&sequential_129_lstm_134_while_identity/sequential_129/lstm_134/while/Identity:output:0*S
_input_shapesB
@: : : : :?????????2:?????????2: ::::: 
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
:?????????2:-)
'
_output_shapes
:?????????2:

_output_shapes
: :

_output_shapes
:
?
?
*__inference_lstm_134_layer_call_fn_1065659

inputs
unknown
	unknown_0
	unknown_1
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????2*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_lstm_134_layer_call_and_return_conditional_losses_10643522
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????22

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????:::22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:?????????
 
_user_specified_nameinputs
?C
?
while_body_1065563
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_08
4while_lstm_cell_134_matmul_readvariableop_resource_0:
6while_lstm_cell_134_matmul_1_readvariableop_resource_09
5while_lstm_cell_134_biasadd_readvariableop_resource_0
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor6
2while_lstm_cell_134_matmul_readvariableop_resource8
4while_lstm_cell_134_matmul_1_readvariableop_resource7
3while_lstm_cell_134_biasadd_readvariableop_resource??*while/lstm_cell_134/BiasAdd/ReadVariableOp?)while/lstm_cell_134/MatMul/ReadVariableOp?+while/lstm_cell_134/MatMul_1/ReadVariableOp?
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   29
7while/TensorArrayV2Read/TensorListGetItem/element_shape?
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:?????????*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItem?
)while/lstm_cell_134/MatMul/ReadVariableOpReadVariableOp4while_lstm_cell_134_matmul_readvariableop_resource_0*
_output_shapes
:	?*
dtype02+
)while/lstm_cell_134/MatMul/ReadVariableOp?
while/lstm_cell_134/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:01while/lstm_cell_134/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_134/MatMul?
+while/lstm_cell_134/MatMul_1/ReadVariableOpReadVariableOp6while_lstm_cell_134_matmul_1_readvariableop_resource_0*
_output_shapes
:	2?*
dtype02-
+while/lstm_cell_134/MatMul_1/ReadVariableOp?
while/lstm_cell_134/MatMul_1MatMulwhile_placeholder_23while/lstm_cell_134/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_134/MatMul_1?
while/lstm_cell_134/addAddV2$while/lstm_cell_134/MatMul:product:0&while/lstm_cell_134/MatMul_1:product:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_134/add?
*while/lstm_cell_134/BiasAdd/ReadVariableOpReadVariableOp5while_lstm_cell_134_biasadd_readvariableop_resource_0*
_output_shapes	
:?*
dtype02,
*while/lstm_cell_134/BiasAdd/ReadVariableOp?
while/lstm_cell_134/BiasAddBiasAddwhile/lstm_cell_134/add:z:02while/lstm_cell_134/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_134/BiasAddx
while/lstm_cell_134/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
while/lstm_cell_134/Const?
#while/lstm_cell_134/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2%
#while/lstm_cell_134/split/split_dim?
while/lstm_cell_134/splitSplit,while/lstm_cell_134/split/split_dim:output:0$while/lstm_cell_134/BiasAdd:output:0*
T0*`
_output_shapesN
L:?????????2:?????????2:?????????2:?????????2*
	num_split2
while/lstm_cell_134/split?
while/lstm_cell_134/SigmoidSigmoid"while/lstm_cell_134/split:output:0*
T0*'
_output_shapes
:?????????22
while/lstm_cell_134/Sigmoid?
while/lstm_cell_134/Sigmoid_1Sigmoid"while/lstm_cell_134/split:output:1*
T0*'
_output_shapes
:?????????22
while/lstm_cell_134/Sigmoid_1?
while/lstm_cell_134/mulMul!while/lstm_cell_134/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:?????????22
while/lstm_cell_134/mul?
while/lstm_cell_134/ReluRelu"while/lstm_cell_134/split:output:2*
T0*'
_output_shapes
:?????????22
while/lstm_cell_134/Relu?
while/lstm_cell_134/mul_1Mulwhile/lstm_cell_134/Sigmoid:y:0&while/lstm_cell_134/Relu:activations:0*
T0*'
_output_shapes
:?????????22
while/lstm_cell_134/mul_1?
while/lstm_cell_134/add_1AddV2while/lstm_cell_134/mul:z:0while/lstm_cell_134/mul_1:z:0*
T0*'
_output_shapes
:?????????22
while/lstm_cell_134/add_1?
while/lstm_cell_134/Sigmoid_2Sigmoid"while/lstm_cell_134/split:output:3*
T0*'
_output_shapes
:?????????22
while/lstm_cell_134/Sigmoid_2?
while/lstm_cell_134/Relu_1Reluwhile/lstm_cell_134/add_1:z:0*
T0*'
_output_shapes
:?????????22
while/lstm_cell_134/Relu_1?
while/lstm_cell_134/mul_2Mul!while/lstm_cell_134/Sigmoid_2:y:0(while/lstm_cell_134/Relu_1:activations:0*
T0*'
_output_shapes
:?????????22
while/lstm_cell_134/mul_2?
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_134/mul_2:z:0*
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
while/add_1?
while/IdentityIdentitywhile/add_1:z:0+^while/lstm_cell_134/BiasAdd/ReadVariableOp*^while/lstm_cell_134/MatMul/ReadVariableOp,^while/lstm_cell_134/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity?
while/Identity_1Identitywhile_while_maximum_iterations+^while/lstm_cell_134/BiasAdd/ReadVariableOp*^while/lstm_cell_134/MatMul/ReadVariableOp,^while/lstm_cell_134/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_1?
while/Identity_2Identitywhile/add:z:0+^while/lstm_cell_134/BiasAdd/ReadVariableOp*^while/lstm_cell_134/MatMul/ReadVariableOp,^while/lstm_cell_134/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_2?
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0+^while/lstm_cell_134/BiasAdd/ReadVariableOp*^while/lstm_cell_134/MatMul/ReadVariableOp,^while/lstm_cell_134/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_3?
while/Identity_4Identitywhile/lstm_cell_134/mul_2:z:0+^while/lstm_cell_134/BiasAdd/ReadVariableOp*^while/lstm_cell_134/MatMul/ReadVariableOp,^while/lstm_cell_134/MatMul_1/ReadVariableOp*
T0*'
_output_shapes
:?????????22
while/Identity_4?
while/Identity_5Identitywhile/lstm_cell_134/add_1:z:0+^while/lstm_cell_134/BiasAdd/ReadVariableOp*^while/lstm_cell_134/MatMul/ReadVariableOp,^while/lstm_cell_134/MatMul_1/ReadVariableOp*
T0*'
_output_shapes
:?????????22
while/Identity_5")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"l
3while_lstm_cell_134_biasadd_readvariableop_resource5while_lstm_cell_134_biasadd_readvariableop_resource_0"n
4while_lstm_cell_134_matmul_1_readvariableop_resource6while_lstm_cell_134_matmul_1_readvariableop_resource_0"j
2while_lstm_cell_134_matmul_readvariableop_resource4while_lstm_cell_134_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"?
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*Q
_input_shapes@
>: : : : :?????????2:?????????2: : :::2X
*while/lstm_cell_134/BiasAdd/ReadVariableOp*while/lstm_cell_134/BiasAdd/ReadVariableOp2V
)while/lstm_cell_134/MatMul/ReadVariableOp)while/lstm_cell_134/MatMul/ReadVariableOp2Z
+while/lstm_cell_134/MatMul_1/ReadVariableOp+while/lstm_cell_134/MatMul_1/ReadVariableOp: 
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
:?????????2:-)
'
_output_shapes
:?????????2:

_output_shapes
: :

_output_shapes
: 
?
?
K__inference_sequential_129_layer_call_and_return_conditional_losses_1064578
lstm_134_input
lstm_134_1064565
lstm_134_1064567
lstm_134_1064569
dense_128_1064572
dense_128_1064574
identity??!dense_128/StatefulPartitionedCall? lstm_134/StatefulPartitionedCall?
 lstm_134/StatefulPartitionedCallStatefulPartitionedCalllstm_134_inputlstm_134_1064565lstm_134_1064567lstm_134_1064569*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????2*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_lstm_134_layer_call_and_return_conditional_losses_10645052"
 lstm_134/StatefulPartitionedCall?
!dense_128/StatefulPartitionedCallStatefulPartitionedCall)lstm_134/StatefulPartitionedCall:output:0dense_128_1064572dense_128_1064574*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_dense_128_layer_call_and_return_conditional_losses_10645452#
!dense_128/StatefulPartitionedCall?
IdentityIdentity*dense_128/StatefulPartitionedCall:output:0"^dense_128/StatefulPartitionedCall!^lstm_134/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:?????????:::::2F
!dense_128/StatefulPartitionedCall!dense_128/StatefulPartitionedCall2D
 lstm_134/StatefulPartitionedCall lstm_134/StatefulPartitionedCall:[ W
+
_output_shapes
:?????????
(
_user_specified_namelstm_134_input
?
?
*__inference_lstm_134_layer_call_fn_1065342
inputs_0
unknown
	unknown_0
	unknown_1
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputs_0unknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????2*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_lstm_134_layer_call_and_return_conditional_losses_10641872
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????22

Identity"
identityIdentity:output:0*?
_input_shapes.
,:??????????????????:::22
StatefulPartitionedCallStatefulPartitionedCall:^ Z
4
_output_shapes"
 :??????????????????
"
_user_specified_name
inputs/0
?%
?
while_body_1063986
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0!
while_lstm_cell_134_1064010_0!
while_lstm_cell_134_1064012_0!
while_lstm_cell_134_1064014_0
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor
while_lstm_cell_134_1064010
while_lstm_cell_134_1064012
while_lstm_cell_134_1064014??+while/lstm_cell_134/StatefulPartitionedCall?
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   29
7while/TensorArrayV2Read/TensorListGetItem/element_shape?
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:?????????*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItem?
+while/lstm_cell_134/StatefulPartitionedCallStatefulPartitionedCall0while/TensorArrayV2Read/TensorListGetItem:item:0while_placeholder_2while_placeholder_3while_lstm_cell_134_1064010_0while_lstm_cell_134_1064012_0while_lstm_cell_134_1064014_0*
Tin

2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:?????????2:?????????2:?????????2*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *S
fNRL
J__inference_lstm_cell_134_layer_call_and_return_conditional_losses_10636592-
+while/lstm_cell_134/StatefulPartitionedCall?
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder4while/lstm_cell_134/StatefulPartitionedCall:output:0*
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
while/add_1?
while/IdentityIdentitywhile/add_1:z:0,^while/lstm_cell_134/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity?
while/Identity_1Identitywhile_while_maximum_iterations,^while/lstm_cell_134/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity_1?
while/Identity_2Identitywhile/add:z:0,^while/lstm_cell_134/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity_2?
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0,^while/lstm_cell_134/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity_3?
while/Identity_4Identity4while/lstm_cell_134/StatefulPartitionedCall:output:1,^while/lstm_cell_134/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????22
while/Identity_4?
while/Identity_5Identity4while/lstm_cell_134/StatefulPartitionedCall:output:2,^while/lstm_cell_134/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????22
while/Identity_5")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"<
while_lstm_cell_134_1064010while_lstm_cell_134_1064010_0"<
while_lstm_cell_134_1064012while_lstm_cell_134_1064012_0"<
while_lstm_cell_134_1064014while_lstm_cell_134_1064014_0"0
while_strided_slice_1while_strided_slice_1_0"?
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*Q
_input_shapes@
>: : : : :?????????2:?????????2: : :::2Z
+while/lstm_cell_134/StatefulPartitionedCall+while/lstm_cell_134/StatefulPartitionedCall: 
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
:?????????2:-)
'
_output_shapes
:?????????2:

_output_shapes
: :

_output_shapes
: 
?
?
while_cond_1064266
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_15
1while_while_cond_1064266___redundant_placeholder05
1while_while_cond_1064266___redundant_placeholder15
1while_while_cond_1064266___redundant_placeholder25
1while_while_cond_1064266___redundant_placeholder3
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
@: : : : :?????????2:?????????2: ::::: 
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
:?????????2:-)
'
_output_shapes
:?????????2:

_output_shapes
: :

_output_shapes
:
?u
?
K__inference_sequential_129_layer_call_and_return_conditional_losses_1064825

inputs9
5lstm_134_lstm_cell_134_matmul_readvariableop_resource;
7lstm_134_lstm_cell_134_matmul_1_readvariableop_resource:
6lstm_134_lstm_cell_134_biasadd_readvariableop_resource,
(dense_128_matmul_readvariableop_resource-
)dense_128_biasadd_readvariableop_resource
identity?? dense_128/BiasAdd/ReadVariableOp?dense_128/MatMul/ReadVariableOp?-lstm_134/lstm_cell_134/BiasAdd/ReadVariableOp?,lstm_134/lstm_cell_134/MatMul/ReadVariableOp?.lstm_134/lstm_cell_134/MatMul_1/ReadVariableOp?lstm_134/whileV
lstm_134/ShapeShapeinputs*
T0*
_output_shapes
:2
lstm_134/Shape?
lstm_134/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
lstm_134/strided_slice/stack?
lstm_134/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2 
lstm_134/strided_slice/stack_1?
lstm_134/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2 
lstm_134/strided_slice/stack_2?
lstm_134/strided_sliceStridedSlicelstm_134/Shape:output:0%lstm_134/strided_slice/stack:output:0'lstm_134/strided_slice/stack_1:output:0'lstm_134/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
lstm_134/strided_slicen
lstm_134/zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :22
lstm_134/zeros/mul/y?
lstm_134/zeros/mulMullstm_134/strided_slice:output:0lstm_134/zeros/mul/y:output:0*
T0*
_output_shapes
: 2
lstm_134/zeros/mulq
lstm_134/zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2
lstm_134/zeros/Less/y?
lstm_134/zeros/LessLesslstm_134/zeros/mul:z:0lstm_134/zeros/Less/y:output:0*
T0*
_output_shapes
: 2
lstm_134/zeros/Lesst
lstm_134/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :22
lstm_134/zeros/packed/1?
lstm_134/zeros/packedPacklstm_134/strided_slice:output:0 lstm_134/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
lstm_134/zeros/packedq
lstm_134/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
lstm_134/zeros/Const?
lstm_134/zerosFilllstm_134/zeros/packed:output:0lstm_134/zeros/Const:output:0*
T0*'
_output_shapes
:?????????22
lstm_134/zerosr
lstm_134/zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B :22
lstm_134/zeros_1/mul/y?
lstm_134/zeros_1/mulMullstm_134/strided_slice:output:0lstm_134/zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
lstm_134/zeros_1/mulu
lstm_134/zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2
lstm_134/zeros_1/Less/y?
lstm_134/zeros_1/LessLesslstm_134/zeros_1/mul:z:0 lstm_134/zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
lstm_134/zeros_1/Lessx
lstm_134/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :22
lstm_134/zeros_1/packed/1?
lstm_134/zeros_1/packedPacklstm_134/strided_slice:output:0"lstm_134/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
lstm_134/zeros_1/packedu
lstm_134/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
lstm_134/zeros_1/Const?
lstm_134/zeros_1Fill lstm_134/zeros_1/packed:output:0lstm_134/zeros_1/Const:output:0*
T0*'
_output_shapes
:?????????22
lstm_134/zeros_1?
lstm_134/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
lstm_134/transpose/perm?
lstm_134/transpose	Transposeinputs lstm_134/transpose/perm:output:0*
T0*+
_output_shapes
:?????????2
lstm_134/transposej
lstm_134/Shape_1Shapelstm_134/transpose:y:0*
T0*
_output_shapes
:2
lstm_134/Shape_1?
lstm_134/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2 
lstm_134/strided_slice_1/stack?
 lstm_134/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2"
 lstm_134/strided_slice_1/stack_1?
 lstm_134/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2"
 lstm_134/strided_slice_1/stack_2?
lstm_134/strided_slice_1StridedSlicelstm_134/Shape_1:output:0'lstm_134/strided_slice_1/stack:output:0)lstm_134/strided_slice_1/stack_1:output:0)lstm_134/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
lstm_134/strided_slice_1?
$lstm_134/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
?????????2&
$lstm_134/TensorArrayV2/element_shape?
lstm_134/TensorArrayV2TensorListReserve-lstm_134/TensorArrayV2/element_shape:output:0!lstm_134/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
lstm_134/TensorArrayV2?
>lstm_134/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   2@
>lstm_134/TensorArrayUnstack/TensorListFromTensor/element_shape?
0lstm_134/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorlstm_134/transpose:y:0Glstm_134/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type022
0lstm_134/TensorArrayUnstack/TensorListFromTensor?
lstm_134/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2 
lstm_134/strided_slice_2/stack?
 lstm_134/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2"
 lstm_134/strided_slice_2/stack_1?
 lstm_134/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2"
 lstm_134/strided_slice_2/stack_2?
lstm_134/strided_slice_2StridedSlicelstm_134/transpose:y:0'lstm_134/strided_slice_2/stack:output:0)lstm_134/strided_slice_2/stack_1:output:0)lstm_134/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*
shrink_axis_mask2
lstm_134/strided_slice_2?
,lstm_134/lstm_cell_134/MatMul/ReadVariableOpReadVariableOp5lstm_134_lstm_cell_134_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype02.
,lstm_134/lstm_cell_134/MatMul/ReadVariableOp?
lstm_134/lstm_cell_134/MatMulMatMul!lstm_134/strided_slice_2:output:04lstm_134/lstm_cell_134/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
lstm_134/lstm_cell_134/MatMul?
.lstm_134/lstm_cell_134/MatMul_1/ReadVariableOpReadVariableOp7lstm_134_lstm_cell_134_matmul_1_readvariableop_resource*
_output_shapes
:	2?*
dtype020
.lstm_134/lstm_cell_134/MatMul_1/ReadVariableOp?
lstm_134/lstm_cell_134/MatMul_1MatMullstm_134/zeros:output:06lstm_134/lstm_cell_134/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2!
lstm_134/lstm_cell_134/MatMul_1?
lstm_134/lstm_cell_134/addAddV2'lstm_134/lstm_cell_134/MatMul:product:0)lstm_134/lstm_cell_134/MatMul_1:product:0*
T0*(
_output_shapes
:??????????2
lstm_134/lstm_cell_134/add?
-lstm_134/lstm_cell_134/BiasAdd/ReadVariableOpReadVariableOp6lstm_134_lstm_cell_134_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02/
-lstm_134/lstm_cell_134/BiasAdd/ReadVariableOp?
lstm_134/lstm_cell_134/BiasAddBiasAddlstm_134/lstm_cell_134/add:z:05lstm_134/lstm_cell_134/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2 
lstm_134/lstm_cell_134/BiasAdd~
lstm_134/lstm_cell_134/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_134/lstm_cell_134/Const?
&lstm_134/lstm_cell_134/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2(
&lstm_134/lstm_cell_134/split/split_dim?
lstm_134/lstm_cell_134/splitSplit/lstm_134/lstm_cell_134/split/split_dim:output:0'lstm_134/lstm_cell_134/BiasAdd:output:0*
T0*`
_output_shapesN
L:?????????2:?????????2:?????????2:?????????2*
	num_split2
lstm_134/lstm_cell_134/split?
lstm_134/lstm_cell_134/SigmoidSigmoid%lstm_134/lstm_cell_134/split:output:0*
T0*'
_output_shapes
:?????????22 
lstm_134/lstm_cell_134/Sigmoid?
 lstm_134/lstm_cell_134/Sigmoid_1Sigmoid%lstm_134/lstm_cell_134/split:output:1*
T0*'
_output_shapes
:?????????22"
 lstm_134/lstm_cell_134/Sigmoid_1?
lstm_134/lstm_cell_134/mulMul$lstm_134/lstm_cell_134/Sigmoid_1:y:0lstm_134/zeros_1:output:0*
T0*'
_output_shapes
:?????????22
lstm_134/lstm_cell_134/mul?
lstm_134/lstm_cell_134/ReluRelu%lstm_134/lstm_cell_134/split:output:2*
T0*'
_output_shapes
:?????????22
lstm_134/lstm_cell_134/Relu?
lstm_134/lstm_cell_134/mul_1Mul"lstm_134/lstm_cell_134/Sigmoid:y:0)lstm_134/lstm_cell_134/Relu:activations:0*
T0*'
_output_shapes
:?????????22
lstm_134/lstm_cell_134/mul_1?
lstm_134/lstm_cell_134/add_1AddV2lstm_134/lstm_cell_134/mul:z:0 lstm_134/lstm_cell_134/mul_1:z:0*
T0*'
_output_shapes
:?????????22
lstm_134/lstm_cell_134/add_1?
 lstm_134/lstm_cell_134/Sigmoid_2Sigmoid%lstm_134/lstm_cell_134/split:output:3*
T0*'
_output_shapes
:?????????22"
 lstm_134/lstm_cell_134/Sigmoid_2?
lstm_134/lstm_cell_134/Relu_1Relu lstm_134/lstm_cell_134/add_1:z:0*
T0*'
_output_shapes
:?????????22
lstm_134/lstm_cell_134/Relu_1?
lstm_134/lstm_cell_134/mul_2Mul$lstm_134/lstm_cell_134/Sigmoid_2:y:0+lstm_134/lstm_cell_134/Relu_1:activations:0*
T0*'
_output_shapes
:?????????22
lstm_134/lstm_cell_134/mul_2?
&lstm_134/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????2   2(
&lstm_134/TensorArrayV2_1/element_shape?
lstm_134/TensorArrayV2_1TensorListReserve/lstm_134/TensorArrayV2_1/element_shape:output:0!lstm_134/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
lstm_134/TensorArrayV2_1`
lstm_134/timeConst*
_output_shapes
: *
dtype0*
value	B : 2
lstm_134/time?
!lstm_134/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????2#
!lstm_134/while/maximum_iterations|
lstm_134/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
lstm_134/while/loop_counter?
lstm_134/whileWhile$lstm_134/while/loop_counter:output:0*lstm_134/while/maximum_iterations:output:0lstm_134/time:output:0!lstm_134/TensorArrayV2_1:handle:0lstm_134/zeros:output:0lstm_134/zeros_1:output:0!lstm_134/strided_slice_1:output:0@lstm_134/TensorArrayUnstack/TensorListFromTensor:output_handle:05lstm_134_lstm_cell_134_matmul_readvariableop_resource7lstm_134_lstm_cell_134_matmul_1_readvariableop_resource6lstm_134_lstm_cell_134_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :?????????2:?????????2: : : : : *%
_read_only_resource_inputs
	
*'
bodyR
lstm_134_while_body_1064734*'
condR
lstm_134_while_cond_1064733*K
output_shapes:
8: : : : :?????????2:?????????2: : : : : *
parallel_iterations 2
lstm_134/while?
9lstm_134/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????2   2;
9lstm_134/TensorArrayV2Stack/TensorListStack/element_shape?
+lstm_134/TensorArrayV2Stack/TensorListStackTensorListStacklstm_134/while:output:3Blstm_134/TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:?????????2*
element_dtype02-
+lstm_134/TensorArrayV2Stack/TensorListStack?
lstm_134/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????2 
lstm_134/strided_slice_3/stack?
 lstm_134/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2"
 lstm_134/strided_slice_3/stack_1?
 lstm_134/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2"
 lstm_134/strided_slice_3/stack_2?
lstm_134/strided_slice_3StridedSlice4lstm_134/TensorArrayV2Stack/TensorListStack:tensor:0'lstm_134/strided_slice_3/stack:output:0)lstm_134/strided_slice_3/stack_1:output:0)lstm_134/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????2*
shrink_axis_mask2
lstm_134/strided_slice_3?
lstm_134/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
lstm_134/transpose_1/perm?
lstm_134/transpose_1	Transpose4lstm_134/TensorArrayV2Stack/TensorListStack:tensor:0"lstm_134/transpose_1/perm:output:0*
T0*+
_output_shapes
:?????????22
lstm_134/transpose_1x
lstm_134/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2
lstm_134/runtime?
dense_128/MatMul/ReadVariableOpReadVariableOp(dense_128_matmul_readvariableop_resource*
_output_shapes

:2*
dtype02!
dense_128/MatMul/ReadVariableOp?
dense_128/MatMulMatMul!lstm_134/strided_slice_3:output:0'dense_128/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_128/MatMul?
 dense_128/BiasAdd/ReadVariableOpReadVariableOp)dense_128_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02"
 dense_128/BiasAdd/ReadVariableOp?
dense_128/BiasAddBiasAdddense_128/MatMul:product:0(dense_128/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_128/BiasAdd?
IdentityIdentitydense_128/BiasAdd:output:0!^dense_128/BiasAdd/ReadVariableOp ^dense_128/MatMul/ReadVariableOp.^lstm_134/lstm_cell_134/BiasAdd/ReadVariableOp-^lstm_134/lstm_cell_134/MatMul/ReadVariableOp/^lstm_134/lstm_cell_134/MatMul_1/ReadVariableOp^lstm_134/while*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:?????????:::::2D
 dense_128/BiasAdd/ReadVariableOp dense_128/BiasAdd/ReadVariableOp2B
dense_128/MatMul/ReadVariableOpdense_128/MatMul/ReadVariableOp2^
-lstm_134/lstm_cell_134/BiasAdd/ReadVariableOp-lstm_134/lstm_cell_134/BiasAdd/ReadVariableOp2\
,lstm_134/lstm_cell_134/MatMul/ReadVariableOp,lstm_134/lstm_cell_134/MatMul/ReadVariableOp2`
.lstm_134/lstm_cell_134/MatMul_1/ReadVariableOp.lstm_134/lstm_cell_134/MatMul_1/ReadVariableOp2 
lstm_134/whilelstm_134/while:S O
+
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
0__inference_sequential_129_layer_call_fn_1065014

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3*
Tin

2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*'
_read_only_resource_inputs	
*-
config_proto

CPU

GPU 2J 8? *T
fORM
K__inference_sequential_129_layer_call_and_return_conditional_losses_10646282
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:?????????:::::22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
/__inference_lstm_cell_134_layer_call_fn_1065772

inputs
states_0
states_1
unknown
	unknown_0
	unknown_1
identity

identity_1

identity_2??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsstates_0states_1unknown	unknown_0	unknown_1*
Tin

2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:?????????2:?????????2:?????????2*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *S
fNRL
J__inference_lstm_cell_134_layer_call_and_return_conditional_losses_10636592
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????22

Identity?

Identity_1Identity StatefulPartitionedCall:output:1^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????22

Identity_1?

Identity_2Identity StatefulPartitionedCall:output:2^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????22

Identity_2"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*X
_input_shapesG
E:?????????:?????????2:?????????2:::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs:QM
'
_output_shapes
:?????????2
"
_user_specified_name
states/0:QM
'
_output_shapes
:?????????2
"
_user_specified_name
states/1
?
?
J__inference_lstm_cell_134_layer_call_and_return_conditional_losses_1065722

inputs
states_0
states_1"
matmul_readvariableop_resource$
 matmul_1_readvariableop_resource#
biasadd_readvariableop_resource
identity

identity_1

identity_2??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?MatMul_1/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	?*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
MatMul?
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource*
_output_shapes
:	2?*
dtype02
MatMul_1/ReadVariableOp|
MatMul_1MatMulstates_0MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2

MatMul_1l
addAddV2MatMul:product:0MatMul_1:product:0*
T0*(
_output_shapes
:??????????2
add?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOpy
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2	
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
split/split_dim?
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*`
_output_shapesN
L:?????????2:?????????2:?????????2:?????????2*
	num_split2
split_
SigmoidSigmoidsplit:output:0*
T0*'
_output_shapes
:?????????22	
Sigmoidc
	Sigmoid_1Sigmoidsplit:output:1*
T0*'
_output_shapes
:?????????22
	Sigmoid_1\
mulMulSigmoid_1:y:0states_1*
T0*'
_output_shapes
:?????????22
mulV
ReluRelusplit:output:2*
T0*'
_output_shapes
:?????????22
Reluh
mul_1MulSigmoid:y:0Relu:activations:0*
T0*'
_output_shapes
:?????????22
mul_1]
add_1AddV2mul:z:0	mul_1:z:0*
T0*'
_output_shapes
:?????????22
add_1c
	Sigmoid_2Sigmoidsplit:output:3*
T0*'
_output_shapes
:?????????22
	Sigmoid_2U
Relu_1Relu	add_1:z:0*
T0*'
_output_shapes
:?????????22
Relu_1l
mul_2MulSigmoid_2:y:0Relu_1:activations:0*
T0*'
_output_shapes
:?????????22
mul_2?
IdentityIdentity	mul_2:z:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*'
_output_shapes
:?????????22

Identity?

Identity_1Identity	mul_2:z:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*'
_output_shapes
:?????????22

Identity_1?

Identity_2Identity	add_1:z:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*'
_output_shapes
:?????????22

Identity_2"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*X
_input_shapesG
E:?????????:?????????2:?????????2:::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs:QM
'
_output_shapes
:?????????2
"
_user_specified_name
states/0:QM
'
_output_shapes
:?????????2
"
_user_specified_name
states/1
?Q
?

lstm_134_while_body_1064734.
*lstm_134_while_lstm_134_while_loop_counter4
0lstm_134_while_lstm_134_while_maximum_iterations
lstm_134_while_placeholder 
lstm_134_while_placeholder_1 
lstm_134_while_placeholder_2 
lstm_134_while_placeholder_3-
)lstm_134_while_lstm_134_strided_slice_1_0i
elstm_134_while_tensorarrayv2read_tensorlistgetitem_lstm_134_tensorarrayunstack_tensorlistfromtensor_0A
=lstm_134_while_lstm_cell_134_matmul_readvariableop_resource_0C
?lstm_134_while_lstm_cell_134_matmul_1_readvariableop_resource_0B
>lstm_134_while_lstm_cell_134_biasadd_readvariableop_resource_0
lstm_134_while_identity
lstm_134_while_identity_1
lstm_134_while_identity_2
lstm_134_while_identity_3
lstm_134_while_identity_4
lstm_134_while_identity_5+
'lstm_134_while_lstm_134_strided_slice_1g
clstm_134_while_tensorarrayv2read_tensorlistgetitem_lstm_134_tensorarrayunstack_tensorlistfromtensor?
;lstm_134_while_lstm_cell_134_matmul_readvariableop_resourceA
=lstm_134_while_lstm_cell_134_matmul_1_readvariableop_resource@
<lstm_134_while_lstm_cell_134_biasadd_readvariableop_resource??3lstm_134/while/lstm_cell_134/BiasAdd/ReadVariableOp?2lstm_134/while/lstm_cell_134/MatMul/ReadVariableOp?4lstm_134/while/lstm_cell_134/MatMul_1/ReadVariableOp?
@lstm_134/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   2B
@lstm_134/while/TensorArrayV2Read/TensorListGetItem/element_shape?
2lstm_134/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemelstm_134_while_tensorarrayv2read_tensorlistgetitem_lstm_134_tensorarrayunstack_tensorlistfromtensor_0lstm_134_while_placeholderIlstm_134/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:?????????*
element_dtype024
2lstm_134/while/TensorArrayV2Read/TensorListGetItem?
2lstm_134/while/lstm_cell_134/MatMul/ReadVariableOpReadVariableOp=lstm_134_while_lstm_cell_134_matmul_readvariableop_resource_0*
_output_shapes
:	?*
dtype024
2lstm_134/while/lstm_cell_134/MatMul/ReadVariableOp?
#lstm_134/while/lstm_cell_134/MatMulMatMul9lstm_134/while/TensorArrayV2Read/TensorListGetItem:item:0:lstm_134/while/lstm_cell_134/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2%
#lstm_134/while/lstm_cell_134/MatMul?
4lstm_134/while/lstm_cell_134/MatMul_1/ReadVariableOpReadVariableOp?lstm_134_while_lstm_cell_134_matmul_1_readvariableop_resource_0*
_output_shapes
:	2?*
dtype026
4lstm_134/while/lstm_cell_134/MatMul_1/ReadVariableOp?
%lstm_134/while/lstm_cell_134/MatMul_1MatMullstm_134_while_placeholder_2<lstm_134/while/lstm_cell_134/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2'
%lstm_134/while/lstm_cell_134/MatMul_1?
 lstm_134/while/lstm_cell_134/addAddV2-lstm_134/while/lstm_cell_134/MatMul:product:0/lstm_134/while/lstm_cell_134/MatMul_1:product:0*
T0*(
_output_shapes
:??????????2"
 lstm_134/while/lstm_cell_134/add?
3lstm_134/while/lstm_cell_134/BiasAdd/ReadVariableOpReadVariableOp>lstm_134_while_lstm_cell_134_biasadd_readvariableop_resource_0*
_output_shapes	
:?*
dtype025
3lstm_134/while/lstm_cell_134/BiasAdd/ReadVariableOp?
$lstm_134/while/lstm_cell_134/BiasAddBiasAdd$lstm_134/while/lstm_cell_134/add:z:0;lstm_134/while/lstm_cell_134/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2&
$lstm_134/while/lstm_cell_134/BiasAdd?
"lstm_134/while/lstm_cell_134/ConstConst*
_output_shapes
: *
dtype0*
value	B :2$
"lstm_134/while/lstm_cell_134/Const?
,lstm_134/while/lstm_cell_134/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2.
,lstm_134/while/lstm_cell_134/split/split_dim?
"lstm_134/while/lstm_cell_134/splitSplit5lstm_134/while/lstm_cell_134/split/split_dim:output:0-lstm_134/while/lstm_cell_134/BiasAdd:output:0*
T0*`
_output_shapesN
L:?????????2:?????????2:?????????2:?????????2*
	num_split2$
"lstm_134/while/lstm_cell_134/split?
$lstm_134/while/lstm_cell_134/SigmoidSigmoid+lstm_134/while/lstm_cell_134/split:output:0*
T0*'
_output_shapes
:?????????22&
$lstm_134/while/lstm_cell_134/Sigmoid?
&lstm_134/while/lstm_cell_134/Sigmoid_1Sigmoid+lstm_134/while/lstm_cell_134/split:output:1*
T0*'
_output_shapes
:?????????22(
&lstm_134/while/lstm_cell_134/Sigmoid_1?
 lstm_134/while/lstm_cell_134/mulMul*lstm_134/while/lstm_cell_134/Sigmoid_1:y:0lstm_134_while_placeholder_3*
T0*'
_output_shapes
:?????????22"
 lstm_134/while/lstm_cell_134/mul?
!lstm_134/while/lstm_cell_134/ReluRelu+lstm_134/while/lstm_cell_134/split:output:2*
T0*'
_output_shapes
:?????????22#
!lstm_134/while/lstm_cell_134/Relu?
"lstm_134/while/lstm_cell_134/mul_1Mul(lstm_134/while/lstm_cell_134/Sigmoid:y:0/lstm_134/while/lstm_cell_134/Relu:activations:0*
T0*'
_output_shapes
:?????????22$
"lstm_134/while/lstm_cell_134/mul_1?
"lstm_134/while/lstm_cell_134/add_1AddV2$lstm_134/while/lstm_cell_134/mul:z:0&lstm_134/while/lstm_cell_134/mul_1:z:0*
T0*'
_output_shapes
:?????????22$
"lstm_134/while/lstm_cell_134/add_1?
&lstm_134/while/lstm_cell_134/Sigmoid_2Sigmoid+lstm_134/while/lstm_cell_134/split:output:3*
T0*'
_output_shapes
:?????????22(
&lstm_134/while/lstm_cell_134/Sigmoid_2?
#lstm_134/while/lstm_cell_134/Relu_1Relu&lstm_134/while/lstm_cell_134/add_1:z:0*
T0*'
_output_shapes
:?????????22%
#lstm_134/while/lstm_cell_134/Relu_1?
"lstm_134/while/lstm_cell_134/mul_2Mul*lstm_134/while/lstm_cell_134/Sigmoid_2:y:01lstm_134/while/lstm_cell_134/Relu_1:activations:0*
T0*'
_output_shapes
:?????????22$
"lstm_134/while/lstm_cell_134/mul_2?
3lstm_134/while/TensorArrayV2Write/TensorListSetItemTensorListSetItemlstm_134_while_placeholder_1lstm_134_while_placeholder&lstm_134/while/lstm_cell_134/mul_2:z:0*
_output_shapes
: *
element_dtype025
3lstm_134/while/TensorArrayV2Write/TensorListSetItemn
lstm_134/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_134/while/add/y?
lstm_134/while/addAddV2lstm_134_while_placeholderlstm_134/while/add/y:output:0*
T0*
_output_shapes
: 2
lstm_134/while/addr
lstm_134/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_134/while/add_1/y?
lstm_134/while/add_1AddV2*lstm_134_while_lstm_134_while_loop_counterlstm_134/while/add_1/y:output:0*
T0*
_output_shapes
: 2
lstm_134/while/add_1?
lstm_134/while/IdentityIdentitylstm_134/while/add_1:z:04^lstm_134/while/lstm_cell_134/BiasAdd/ReadVariableOp3^lstm_134/while/lstm_cell_134/MatMul/ReadVariableOp5^lstm_134/while/lstm_cell_134/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
lstm_134/while/Identity?
lstm_134/while/Identity_1Identity0lstm_134_while_lstm_134_while_maximum_iterations4^lstm_134/while/lstm_cell_134/BiasAdd/ReadVariableOp3^lstm_134/while/lstm_cell_134/MatMul/ReadVariableOp5^lstm_134/while/lstm_cell_134/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
lstm_134/while/Identity_1?
lstm_134/while/Identity_2Identitylstm_134/while/add:z:04^lstm_134/while/lstm_cell_134/BiasAdd/ReadVariableOp3^lstm_134/while/lstm_cell_134/MatMul/ReadVariableOp5^lstm_134/while/lstm_cell_134/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
lstm_134/while/Identity_2?
lstm_134/while/Identity_3IdentityClstm_134/while/TensorArrayV2Write/TensorListSetItem:output_handle:04^lstm_134/while/lstm_cell_134/BiasAdd/ReadVariableOp3^lstm_134/while/lstm_cell_134/MatMul/ReadVariableOp5^lstm_134/while/lstm_cell_134/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
lstm_134/while/Identity_3?
lstm_134/while/Identity_4Identity&lstm_134/while/lstm_cell_134/mul_2:z:04^lstm_134/while/lstm_cell_134/BiasAdd/ReadVariableOp3^lstm_134/while/lstm_cell_134/MatMul/ReadVariableOp5^lstm_134/while/lstm_cell_134/MatMul_1/ReadVariableOp*
T0*'
_output_shapes
:?????????22
lstm_134/while/Identity_4?
lstm_134/while/Identity_5Identity&lstm_134/while/lstm_cell_134/add_1:z:04^lstm_134/while/lstm_cell_134/BiasAdd/ReadVariableOp3^lstm_134/while/lstm_cell_134/MatMul/ReadVariableOp5^lstm_134/while/lstm_cell_134/MatMul_1/ReadVariableOp*
T0*'
_output_shapes
:?????????22
lstm_134/while/Identity_5";
lstm_134_while_identity lstm_134/while/Identity:output:0"?
lstm_134_while_identity_1"lstm_134/while/Identity_1:output:0"?
lstm_134_while_identity_2"lstm_134/while/Identity_2:output:0"?
lstm_134_while_identity_3"lstm_134/while/Identity_3:output:0"?
lstm_134_while_identity_4"lstm_134/while/Identity_4:output:0"?
lstm_134_while_identity_5"lstm_134/while/Identity_5:output:0"T
'lstm_134_while_lstm_134_strided_slice_1)lstm_134_while_lstm_134_strided_slice_1_0"~
<lstm_134_while_lstm_cell_134_biasadd_readvariableop_resource>lstm_134_while_lstm_cell_134_biasadd_readvariableop_resource_0"?
=lstm_134_while_lstm_cell_134_matmul_1_readvariableop_resource?lstm_134_while_lstm_cell_134_matmul_1_readvariableop_resource_0"|
;lstm_134_while_lstm_cell_134_matmul_readvariableop_resource=lstm_134_while_lstm_cell_134_matmul_readvariableop_resource_0"?
clstm_134_while_tensorarrayv2read_tensorlistgetitem_lstm_134_tensorarrayunstack_tensorlistfromtensorelstm_134_while_tensorarrayv2read_tensorlistgetitem_lstm_134_tensorarrayunstack_tensorlistfromtensor_0*Q
_input_shapes@
>: : : : :?????????2:?????????2: : :::2j
3lstm_134/while/lstm_cell_134/BiasAdd/ReadVariableOp3lstm_134/while/lstm_cell_134/BiasAdd/ReadVariableOp2h
2lstm_134/while/lstm_cell_134/MatMul/ReadVariableOp2lstm_134/while/lstm_cell_134/MatMul/ReadVariableOp2l
4lstm_134/while/lstm_cell_134/MatMul_1/ReadVariableOp4lstm_134/while/lstm_cell_134/MatMul_1/ReadVariableOp: 
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
:?????????2:-)
'
_output_shapes
:?????????2:

_output_shapes
: :

_output_shapes
: 
?
?
while_cond_1063985
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_15
1while_while_cond_1063985___redundant_placeholder05
1while_while_cond_1063985___redundant_placeholder15
1while_while_cond_1063985___redundant_placeholder25
1while_while_cond_1063985___redundant_placeholder3
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
@: : : : :?????????2:?????????2: ::::: 
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
:?????????2:-)
'
_output_shapes
:?????????2:

_output_shapes
: :

_output_shapes
:
?[
?
E__inference_lstm_134_layer_call_and_return_conditional_losses_1065648

inputs0
,lstm_cell_134_matmul_readvariableop_resource2
.lstm_cell_134_matmul_1_readvariableop_resource1
-lstm_cell_134_biasadd_readvariableop_resource
identity??$lstm_cell_134/BiasAdd/ReadVariableOp?#lstm_cell_134/MatMul/ReadVariableOp?%lstm_cell_134/MatMul_1/ReadVariableOp?whileD
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
strided_slice/stack_2?
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
value	B :22
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
B :?2
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
value	B :22
zeros/packed/1?
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
:?????????22
zeros`
zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B :22
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
B :?2
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
value	B :22
zeros_1/packed/1?
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
:?????????22	
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
:?????????2
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
strided_slice_1/stack_2?
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1?
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
TensorArrayV2/element_shape?
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2?
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   27
5TensorArrayUnstack/TensorListFromTensor/element_shape?
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
strided_slice_2/stack_2?
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*
shrink_axis_mask2
strided_slice_2?
#lstm_cell_134/MatMul/ReadVariableOpReadVariableOp,lstm_cell_134_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype02%
#lstm_cell_134/MatMul/ReadVariableOp?
lstm_cell_134/MatMulMatMulstrided_slice_2:output:0+lstm_cell_134/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
lstm_cell_134/MatMul?
%lstm_cell_134/MatMul_1/ReadVariableOpReadVariableOp.lstm_cell_134_matmul_1_readvariableop_resource*
_output_shapes
:	2?*
dtype02'
%lstm_cell_134/MatMul_1/ReadVariableOp?
lstm_cell_134/MatMul_1MatMulzeros:output:0-lstm_cell_134/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
lstm_cell_134/MatMul_1?
lstm_cell_134/addAddV2lstm_cell_134/MatMul:product:0 lstm_cell_134/MatMul_1:product:0*
T0*(
_output_shapes
:??????????2
lstm_cell_134/add?
$lstm_cell_134/BiasAdd/ReadVariableOpReadVariableOp-lstm_cell_134_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02&
$lstm_cell_134/BiasAdd/ReadVariableOp?
lstm_cell_134/BiasAddBiasAddlstm_cell_134/add:z:0,lstm_cell_134/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
lstm_cell_134/BiasAddl
lstm_cell_134/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_134/Const?
lstm_cell_134/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_134/split/split_dim?
lstm_cell_134/splitSplit&lstm_cell_134/split/split_dim:output:0lstm_cell_134/BiasAdd:output:0*
T0*`
_output_shapesN
L:?????????2:?????????2:?????????2:?????????2*
	num_split2
lstm_cell_134/split?
lstm_cell_134/SigmoidSigmoidlstm_cell_134/split:output:0*
T0*'
_output_shapes
:?????????22
lstm_cell_134/Sigmoid?
lstm_cell_134/Sigmoid_1Sigmoidlstm_cell_134/split:output:1*
T0*'
_output_shapes
:?????????22
lstm_cell_134/Sigmoid_1?
lstm_cell_134/mulMullstm_cell_134/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:?????????22
lstm_cell_134/mul?
lstm_cell_134/ReluRelulstm_cell_134/split:output:2*
T0*'
_output_shapes
:?????????22
lstm_cell_134/Relu?
lstm_cell_134/mul_1Mullstm_cell_134/Sigmoid:y:0 lstm_cell_134/Relu:activations:0*
T0*'
_output_shapes
:?????????22
lstm_cell_134/mul_1?
lstm_cell_134/add_1AddV2lstm_cell_134/mul:z:0lstm_cell_134/mul_1:z:0*
T0*'
_output_shapes
:?????????22
lstm_cell_134/add_1?
lstm_cell_134/Sigmoid_2Sigmoidlstm_cell_134/split:output:3*
T0*'
_output_shapes
:?????????22
lstm_cell_134/Sigmoid_2
lstm_cell_134/Relu_1Relulstm_cell_134/add_1:z:0*
T0*'
_output_shapes
:?????????22
lstm_cell_134/Relu_1?
lstm_cell_134/mul_2Mullstm_cell_134/Sigmoid_2:y:0"lstm_cell_134/Relu_1:activations:0*
T0*'
_output_shapes
:?????????22
lstm_cell_134/mul_2?
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????2   2
TensorArrayV2_1/element_shape?
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
?????????2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counter?
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0,lstm_cell_134_matmul_readvariableop_resource.lstm_cell_134_matmul_1_readvariableop_resource-lstm_cell_134_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :?????????2:?????????2: : : : : *%
_read_only_resource_inputs
	
*
bodyR
while_body_1065563*
condR
while_cond_1065562*K
output_shapes:
8: : : : :?????????2:?????????2: : : : : *
parallel_iterations 2
while?
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????2   22
0TensorArrayV2Stack/TensorListStack/element_shape?
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:?????????2*
element_dtype02$
"TensorArrayV2Stack/TensorListStack?
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????2
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
strided_slice_3/stack_2?
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????2*
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/perm?
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*+
_output_shapes
:?????????22
transpose_1f
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2	
runtime?
IdentityIdentitystrided_slice_3:output:0%^lstm_cell_134/BiasAdd/ReadVariableOp$^lstm_cell_134/MatMul/ReadVariableOp&^lstm_cell_134/MatMul_1/ReadVariableOp^while*
T0*'
_output_shapes
:?????????22

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????:::2L
$lstm_cell_134/BiasAdd/ReadVariableOp$lstm_cell_134/BiasAdd/ReadVariableOp2J
#lstm_cell_134/MatMul/ReadVariableOp#lstm_cell_134/MatMul/ReadVariableOp2N
%lstm_cell_134/MatMul_1/ReadVariableOp%lstm_cell_134/MatMul_1/ReadVariableOp2
whilewhile:S O
+
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
0__inference_sequential_129_layer_call_fn_1064641
lstm_134_input
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCalllstm_134_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3*
Tin

2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*'
_read_only_resource_inputs	
*-
config_proto

CPU

GPU 2J 8? *T
fORM
K__inference_sequential_129_layer_call_and_return_conditional_losses_10646282
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:?????????:::::22
StatefulPartitionedCallStatefulPartitionedCall:[ W
+
_output_shapes
:?????????
(
_user_specified_namelstm_134_input
?
?
while_cond_1065081
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_15
1while_while_cond_1065081___redundant_placeholder05
1while_while_cond_1065081___redundant_placeholder15
1while_while_cond_1065081___redundant_placeholder25
1while_while_cond_1065081___redundant_placeholder3
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
@: : : : :?????????2:?????????2: ::::: 
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
:?????????2:-)
'
_output_shapes
:?????????2:

_output_shapes
: :

_output_shapes
:
?C
?
while_body_1065082
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_08
4while_lstm_cell_134_matmul_readvariableop_resource_0:
6while_lstm_cell_134_matmul_1_readvariableop_resource_09
5while_lstm_cell_134_biasadd_readvariableop_resource_0
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor6
2while_lstm_cell_134_matmul_readvariableop_resource8
4while_lstm_cell_134_matmul_1_readvariableop_resource7
3while_lstm_cell_134_biasadd_readvariableop_resource??*while/lstm_cell_134/BiasAdd/ReadVariableOp?)while/lstm_cell_134/MatMul/ReadVariableOp?+while/lstm_cell_134/MatMul_1/ReadVariableOp?
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   29
7while/TensorArrayV2Read/TensorListGetItem/element_shape?
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:?????????*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItem?
)while/lstm_cell_134/MatMul/ReadVariableOpReadVariableOp4while_lstm_cell_134_matmul_readvariableop_resource_0*
_output_shapes
:	?*
dtype02+
)while/lstm_cell_134/MatMul/ReadVariableOp?
while/lstm_cell_134/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:01while/lstm_cell_134/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_134/MatMul?
+while/lstm_cell_134/MatMul_1/ReadVariableOpReadVariableOp6while_lstm_cell_134_matmul_1_readvariableop_resource_0*
_output_shapes
:	2?*
dtype02-
+while/lstm_cell_134/MatMul_1/ReadVariableOp?
while/lstm_cell_134/MatMul_1MatMulwhile_placeholder_23while/lstm_cell_134/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_134/MatMul_1?
while/lstm_cell_134/addAddV2$while/lstm_cell_134/MatMul:product:0&while/lstm_cell_134/MatMul_1:product:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_134/add?
*while/lstm_cell_134/BiasAdd/ReadVariableOpReadVariableOp5while_lstm_cell_134_biasadd_readvariableop_resource_0*
_output_shapes	
:?*
dtype02,
*while/lstm_cell_134/BiasAdd/ReadVariableOp?
while/lstm_cell_134/BiasAddBiasAddwhile/lstm_cell_134/add:z:02while/lstm_cell_134/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_134/BiasAddx
while/lstm_cell_134/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
while/lstm_cell_134/Const?
#while/lstm_cell_134/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2%
#while/lstm_cell_134/split/split_dim?
while/lstm_cell_134/splitSplit,while/lstm_cell_134/split/split_dim:output:0$while/lstm_cell_134/BiasAdd:output:0*
T0*`
_output_shapesN
L:?????????2:?????????2:?????????2:?????????2*
	num_split2
while/lstm_cell_134/split?
while/lstm_cell_134/SigmoidSigmoid"while/lstm_cell_134/split:output:0*
T0*'
_output_shapes
:?????????22
while/lstm_cell_134/Sigmoid?
while/lstm_cell_134/Sigmoid_1Sigmoid"while/lstm_cell_134/split:output:1*
T0*'
_output_shapes
:?????????22
while/lstm_cell_134/Sigmoid_1?
while/lstm_cell_134/mulMul!while/lstm_cell_134/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:?????????22
while/lstm_cell_134/mul?
while/lstm_cell_134/ReluRelu"while/lstm_cell_134/split:output:2*
T0*'
_output_shapes
:?????????22
while/lstm_cell_134/Relu?
while/lstm_cell_134/mul_1Mulwhile/lstm_cell_134/Sigmoid:y:0&while/lstm_cell_134/Relu:activations:0*
T0*'
_output_shapes
:?????????22
while/lstm_cell_134/mul_1?
while/lstm_cell_134/add_1AddV2while/lstm_cell_134/mul:z:0while/lstm_cell_134/mul_1:z:0*
T0*'
_output_shapes
:?????????22
while/lstm_cell_134/add_1?
while/lstm_cell_134/Sigmoid_2Sigmoid"while/lstm_cell_134/split:output:3*
T0*'
_output_shapes
:?????????22
while/lstm_cell_134/Sigmoid_2?
while/lstm_cell_134/Relu_1Reluwhile/lstm_cell_134/add_1:z:0*
T0*'
_output_shapes
:?????????22
while/lstm_cell_134/Relu_1?
while/lstm_cell_134/mul_2Mul!while/lstm_cell_134/Sigmoid_2:y:0(while/lstm_cell_134/Relu_1:activations:0*
T0*'
_output_shapes
:?????????22
while/lstm_cell_134/mul_2?
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_134/mul_2:z:0*
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
while/add_1?
while/IdentityIdentitywhile/add_1:z:0+^while/lstm_cell_134/BiasAdd/ReadVariableOp*^while/lstm_cell_134/MatMul/ReadVariableOp,^while/lstm_cell_134/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity?
while/Identity_1Identitywhile_while_maximum_iterations+^while/lstm_cell_134/BiasAdd/ReadVariableOp*^while/lstm_cell_134/MatMul/ReadVariableOp,^while/lstm_cell_134/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_1?
while/Identity_2Identitywhile/add:z:0+^while/lstm_cell_134/BiasAdd/ReadVariableOp*^while/lstm_cell_134/MatMul/ReadVariableOp,^while/lstm_cell_134/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_2?
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0+^while/lstm_cell_134/BiasAdd/ReadVariableOp*^while/lstm_cell_134/MatMul/ReadVariableOp,^while/lstm_cell_134/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_3?
while/Identity_4Identitywhile/lstm_cell_134/mul_2:z:0+^while/lstm_cell_134/BiasAdd/ReadVariableOp*^while/lstm_cell_134/MatMul/ReadVariableOp,^while/lstm_cell_134/MatMul_1/ReadVariableOp*
T0*'
_output_shapes
:?????????22
while/Identity_4?
while/Identity_5Identitywhile/lstm_cell_134/add_1:z:0+^while/lstm_cell_134/BiasAdd/ReadVariableOp*^while/lstm_cell_134/MatMul/ReadVariableOp,^while/lstm_cell_134/MatMul_1/ReadVariableOp*
T0*'
_output_shapes
:?????????22
while/Identity_5")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"l
3while_lstm_cell_134_biasadd_readvariableop_resource5while_lstm_cell_134_biasadd_readvariableop_resource_0"n
4while_lstm_cell_134_matmul_1_readvariableop_resource6while_lstm_cell_134_matmul_1_readvariableop_resource_0"j
2while_lstm_cell_134_matmul_readvariableop_resource4while_lstm_cell_134_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"?
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*Q
_input_shapes@
>: : : : :?????????2:?????????2: : :::2X
*while/lstm_cell_134/BiasAdd/ReadVariableOp*while/lstm_cell_134/BiasAdd/ReadVariableOp2V
)while/lstm_cell_134/MatMul/ReadVariableOp)while/lstm_cell_134/MatMul/ReadVariableOp2Z
+while/lstm_cell_134/MatMul_1/ReadVariableOp+while/lstm_cell_134/MatMul_1/ReadVariableOp: 
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
:?????????2:-)
'
_output_shapes
:?????????2:

_output_shapes
: :

_output_shapes
: 
?
?
K__inference_sequential_129_layer_call_and_return_conditional_losses_1064597

inputs
lstm_134_1064584
lstm_134_1064586
lstm_134_1064588
dense_128_1064591
dense_128_1064593
identity??!dense_128/StatefulPartitionedCall? lstm_134/StatefulPartitionedCall?
 lstm_134/StatefulPartitionedCallStatefulPartitionedCallinputslstm_134_1064584lstm_134_1064586lstm_134_1064588*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????2*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_lstm_134_layer_call_and_return_conditional_losses_10643522"
 lstm_134/StatefulPartitionedCall?
!dense_128/StatefulPartitionedCallStatefulPartitionedCall)lstm_134/StatefulPartitionedCall:output:0dense_128_1064591dense_128_1064593*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_dense_128_layer_call_and_return_conditional_losses_10645452#
!dense_128/StatefulPartitionedCall?
IdentityIdentity*dense_128/StatefulPartitionedCall:output:0"^dense_128/StatefulPartitionedCall!^lstm_134/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:?????????:::::2F
!dense_128/StatefulPartitionedCall!dense_128/StatefulPartitionedCall2D
 lstm_134/StatefulPartitionedCall lstm_134/StatefulPartitionedCall:S O
+
_output_shapes
:?????????
 
_user_specified_nameinputs
?[
?
E__inference_lstm_134_layer_call_and_return_conditional_losses_1064505

inputs0
,lstm_cell_134_matmul_readvariableop_resource2
.lstm_cell_134_matmul_1_readvariableop_resource1
-lstm_cell_134_biasadd_readvariableop_resource
identity??$lstm_cell_134/BiasAdd/ReadVariableOp?#lstm_cell_134/MatMul/ReadVariableOp?%lstm_cell_134/MatMul_1/ReadVariableOp?whileD
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
strided_slice/stack_2?
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
value	B :22
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
B :?2
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
value	B :22
zeros/packed/1?
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
:?????????22
zeros`
zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B :22
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
B :?2
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
value	B :22
zeros_1/packed/1?
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
:?????????22	
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
:?????????2
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
strided_slice_1/stack_2?
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1?
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
TensorArrayV2/element_shape?
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2?
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   27
5TensorArrayUnstack/TensorListFromTensor/element_shape?
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
strided_slice_2/stack_2?
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*
shrink_axis_mask2
strided_slice_2?
#lstm_cell_134/MatMul/ReadVariableOpReadVariableOp,lstm_cell_134_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype02%
#lstm_cell_134/MatMul/ReadVariableOp?
lstm_cell_134/MatMulMatMulstrided_slice_2:output:0+lstm_cell_134/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
lstm_cell_134/MatMul?
%lstm_cell_134/MatMul_1/ReadVariableOpReadVariableOp.lstm_cell_134_matmul_1_readvariableop_resource*
_output_shapes
:	2?*
dtype02'
%lstm_cell_134/MatMul_1/ReadVariableOp?
lstm_cell_134/MatMul_1MatMulzeros:output:0-lstm_cell_134/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
lstm_cell_134/MatMul_1?
lstm_cell_134/addAddV2lstm_cell_134/MatMul:product:0 lstm_cell_134/MatMul_1:product:0*
T0*(
_output_shapes
:??????????2
lstm_cell_134/add?
$lstm_cell_134/BiasAdd/ReadVariableOpReadVariableOp-lstm_cell_134_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02&
$lstm_cell_134/BiasAdd/ReadVariableOp?
lstm_cell_134/BiasAddBiasAddlstm_cell_134/add:z:0,lstm_cell_134/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
lstm_cell_134/BiasAddl
lstm_cell_134/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_134/Const?
lstm_cell_134/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_134/split/split_dim?
lstm_cell_134/splitSplit&lstm_cell_134/split/split_dim:output:0lstm_cell_134/BiasAdd:output:0*
T0*`
_output_shapesN
L:?????????2:?????????2:?????????2:?????????2*
	num_split2
lstm_cell_134/split?
lstm_cell_134/SigmoidSigmoidlstm_cell_134/split:output:0*
T0*'
_output_shapes
:?????????22
lstm_cell_134/Sigmoid?
lstm_cell_134/Sigmoid_1Sigmoidlstm_cell_134/split:output:1*
T0*'
_output_shapes
:?????????22
lstm_cell_134/Sigmoid_1?
lstm_cell_134/mulMullstm_cell_134/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:?????????22
lstm_cell_134/mul?
lstm_cell_134/ReluRelulstm_cell_134/split:output:2*
T0*'
_output_shapes
:?????????22
lstm_cell_134/Relu?
lstm_cell_134/mul_1Mullstm_cell_134/Sigmoid:y:0 lstm_cell_134/Relu:activations:0*
T0*'
_output_shapes
:?????????22
lstm_cell_134/mul_1?
lstm_cell_134/add_1AddV2lstm_cell_134/mul:z:0lstm_cell_134/mul_1:z:0*
T0*'
_output_shapes
:?????????22
lstm_cell_134/add_1?
lstm_cell_134/Sigmoid_2Sigmoidlstm_cell_134/split:output:3*
T0*'
_output_shapes
:?????????22
lstm_cell_134/Sigmoid_2
lstm_cell_134/Relu_1Relulstm_cell_134/add_1:z:0*
T0*'
_output_shapes
:?????????22
lstm_cell_134/Relu_1?
lstm_cell_134/mul_2Mullstm_cell_134/Sigmoid_2:y:0"lstm_cell_134/Relu_1:activations:0*
T0*'
_output_shapes
:?????????22
lstm_cell_134/mul_2?
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????2   2
TensorArrayV2_1/element_shape?
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
?????????2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counter?
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0,lstm_cell_134_matmul_readvariableop_resource.lstm_cell_134_matmul_1_readvariableop_resource-lstm_cell_134_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :?????????2:?????????2: : : : : *%
_read_only_resource_inputs
	
*
bodyR
while_body_1064420*
condR
while_cond_1064419*K
output_shapes:
8: : : : :?????????2:?????????2: : : : : *
parallel_iterations 2
while?
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????2   22
0TensorArrayV2Stack/TensorListStack/element_shape?
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:?????????2*
element_dtype02$
"TensorArrayV2Stack/TensorListStack?
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????2
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
strided_slice_3/stack_2?
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????2*
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/perm?
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*+
_output_shapes
:?????????22
transpose_1f
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2	
runtime?
IdentityIdentitystrided_slice_3:output:0%^lstm_cell_134/BiasAdd/ReadVariableOp$^lstm_cell_134/MatMul/ReadVariableOp&^lstm_cell_134/MatMul_1/ReadVariableOp^while*
T0*'
_output_shapes
:?????????22

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????:::2L
$lstm_cell_134/BiasAdd/ReadVariableOp$lstm_cell_134/BiasAdd/ReadVariableOp2J
#lstm_cell_134/MatMul/ReadVariableOp#lstm_cell_134/MatMul/ReadVariableOp2N
%lstm_cell_134/MatMul_1/ReadVariableOp%lstm_cell_134/MatMul_1/ReadVariableOp2
whilewhile:S O
+
_output_shapes
:?????????
 
_user_specified_nameinputs
?Q
?

lstm_134_while_body_1064893.
*lstm_134_while_lstm_134_while_loop_counter4
0lstm_134_while_lstm_134_while_maximum_iterations
lstm_134_while_placeholder 
lstm_134_while_placeholder_1 
lstm_134_while_placeholder_2 
lstm_134_while_placeholder_3-
)lstm_134_while_lstm_134_strided_slice_1_0i
elstm_134_while_tensorarrayv2read_tensorlistgetitem_lstm_134_tensorarrayunstack_tensorlistfromtensor_0A
=lstm_134_while_lstm_cell_134_matmul_readvariableop_resource_0C
?lstm_134_while_lstm_cell_134_matmul_1_readvariableop_resource_0B
>lstm_134_while_lstm_cell_134_biasadd_readvariableop_resource_0
lstm_134_while_identity
lstm_134_while_identity_1
lstm_134_while_identity_2
lstm_134_while_identity_3
lstm_134_while_identity_4
lstm_134_while_identity_5+
'lstm_134_while_lstm_134_strided_slice_1g
clstm_134_while_tensorarrayv2read_tensorlistgetitem_lstm_134_tensorarrayunstack_tensorlistfromtensor?
;lstm_134_while_lstm_cell_134_matmul_readvariableop_resourceA
=lstm_134_while_lstm_cell_134_matmul_1_readvariableop_resource@
<lstm_134_while_lstm_cell_134_biasadd_readvariableop_resource??3lstm_134/while/lstm_cell_134/BiasAdd/ReadVariableOp?2lstm_134/while/lstm_cell_134/MatMul/ReadVariableOp?4lstm_134/while/lstm_cell_134/MatMul_1/ReadVariableOp?
@lstm_134/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   2B
@lstm_134/while/TensorArrayV2Read/TensorListGetItem/element_shape?
2lstm_134/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemelstm_134_while_tensorarrayv2read_tensorlistgetitem_lstm_134_tensorarrayunstack_tensorlistfromtensor_0lstm_134_while_placeholderIlstm_134/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:?????????*
element_dtype024
2lstm_134/while/TensorArrayV2Read/TensorListGetItem?
2lstm_134/while/lstm_cell_134/MatMul/ReadVariableOpReadVariableOp=lstm_134_while_lstm_cell_134_matmul_readvariableop_resource_0*
_output_shapes
:	?*
dtype024
2lstm_134/while/lstm_cell_134/MatMul/ReadVariableOp?
#lstm_134/while/lstm_cell_134/MatMulMatMul9lstm_134/while/TensorArrayV2Read/TensorListGetItem:item:0:lstm_134/while/lstm_cell_134/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2%
#lstm_134/while/lstm_cell_134/MatMul?
4lstm_134/while/lstm_cell_134/MatMul_1/ReadVariableOpReadVariableOp?lstm_134_while_lstm_cell_134_matmul_1_readvariableop_resource_0*
_output_shapes
:	2?*
dtype026
4lstm_134/while/lstm_cell_134/MatMul_1/ReadVariableOp?
%lstm_134/while/lstm_cell_134/MatMul_1MatMullstm_134_while_placeholder_2<lstm_134/while/lstm_cell_134/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2'
%lstm_134/while/lstm_cell_134/MatMul_1?
 lstm_134/while/lstm_cell_134/addAddV2-lstm_134/while/lstm_cell_134/MatMul:product:0/lstm_134/while/lstm_cell_134/MatMul_1:product:0*
T0*(
_output_shapes
:??????????2"
 lstm_134/while/lstm_cell_134/add?
3lstm_134/while/lstm_cell_134/BiasAdd/ReadVariableOpReadVariableOp>lstm_134_while_lstm_cell_134_biasadd_readvariableop_resource_0*
_output_shapes	
:?*
dtype025
3lstm_134/while/lstm_cell_134/BiasAdd/ReadVariableOp?
$lstm_134/while/lstm_cell_134/BiasAddBiasAdd$lstm_134/while/lstm_cell_134/add:z:0;lstm_134/while/lstm_cell_134/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2&
$lstm_134/while/lstm_cell_134/BiasAdd?
"lstm_134/while/lstm_cell_134/ConstConst*
_output_shapes
: *
dtype0*
value	B :2$
"lstm_134/while/lstm_cell_134/Const?
,lstm_134/while/lstm_cell_134/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2.
,lstm_134/while/lstm_cell_134/split/split_dim?
"lstm_134/while/lstm_cell_134/splitSplit5lstm_134/while/lstm_cell_134/split/split_dim:output:0-lstm_134/while/lstm_cell_134/BiasAdd:output:0*
T0*`
_output_shapesN
L:?????????2:?????????2:?????????2:?????????2*
	num_split2$
"lstm_134/while/lstm_cell_134/split?
$lstm_134/while/lstm_cell_134/SigmoidSigmoid+lstm_134/while/lstm_cell_134/split:output:0*
T0*'
_output_shapes
:?????????22&
$lstm_134/while/lstm_cell_134/Sigmoid?
&lstm_134/while/lstm_cell_134/Sigmoid_1Sigmoid+lstm_134/while/lstm_cell_134/split:output:1*
T0*'
_output_shapes
:?????????22(
&lstm_134/while/lstm_cell_134/Sigmoid_1?
 lstm_134/while/lstm_cell_134/mulMul*lstm_134/while/lstm_cell_134/Sigmoid_1:y:0lstm_134_while_placeholder_3*
T0*'
_output_shapes
:?????????22"
 lstm_134/while/lstm_cell_134/mul?
!lstm_134/while/lstm_cell_134/ReluRelu+lstm_134/while/lstm_cell_134/split:output:2*
T0*'
_output_shapes
:?????????22#
!lstm_134/while/lstm_cell_134/Relu?
"lstm_134/while/lstm_cell_134/mul_1Mul(lstm_134/while/lstm_cell_134/Sigmoid:y:0/lstm_134/while/lstm_cell_134/Relu:activations:0*
T0*'
_output_shapes
:?????????22$
"lstm_134/while/lstm_cell_134/mul_1?
"lstm_134/while/lstm_cell_134/add_1AddV2$lstm_134/while/lstm_cell_134/mul:z:0&lstm_134/while/lstm_cell_134/mul_1:z:0*
T0*'
_output_shapes
:?????????22$
"lstm_134/while/lstm_cell_134/add_1?
&lstm_134/while/lstm_cell_134/Sigmoid_2Sigmoid+lstm_134/while/lstm_cell_134/split:output:3*
T0*'
_output_shapes
:?????????22(
&lstm_134/while/lstm_cell_134/Sigmoid_2?
#lstm_134/while/lstm_cell_134/Relu_1Relu&lstm_134/while/lstm_cell_134/add_1:z:0*
T0*'
_output_shapes
:?????????22%
#lstm_134/while/lstm_cell_134/Relu_1?
"lstm_134/while/lstm_cell_134/mul_2Mul*lstm_134/while/lstm_cell_134/Sigmoid_2:y:01lstm_134/while/lstm_cell_134/Relu_1:activations:0*
T0*'
_output_shapes
:?????????22$
"lstm_134/while/lstm_cell_134/mul_2?
3lstm_134/while/TensorArrayV2Write/TensorListSetItemTensorListSetItemlstm_134_while_placeholder_1lstm_134_while_placeholder&lstm_134/while/lstm_cell_134/mul_2:z:0*
_output_shapes
: *
element_dtype025
3lstm_134/while/TensorArrayV2Write/TensorListSetItemn
lstm_134/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_134/while/add/y?
lstm_134/while/addAddV2lstm_134_while_placeholderlstm_134/while/add/y:output:0*
T0*
_output_shapes
: 2
lstm_134/while/addr
lstm_134/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_134/while/add_1/y?
lstm_134/while/add_1AddV2*lstm_134_while_lstm_134_while_loop_counterlstm_134/while/add_1/y:output:0*
T0*
_output_shapes
: 2
lstm_134/while/add_1?
lstm_134/while/IdentityIdentitylstm_134/while/add_1:z:04^lstm_134/while/lstm_cell_134/BiasAdd/ReadVariableOp3^lstm_134/while/lstm_cell_134/MatMul/ReadVariableOp5^lstm_134/while/lstm_cell_134/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
lstm_134/while/Identity?
lstm_134/while/Identity_1Identity0lstm_134_while_lstm_134_while_maximum_iterations4^lstm_134/while/lstm_cell_134/BiasAdd/ReadVariableOp3^lstm_134/while/lstm_cell_134/MatMul/ReadVariableOp5^lstm_134/while/lstm_cell_134/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
lstm_134/while/Identity_1?
lstm_134/while/Identity_2Identitylstm_134/while/add:z:04^lstm_134/while/lstm_cell_134/BiasAdd/ReadVariableOp3^lstm_134/while/lstm_cell_134/MatMul/ReadVariableOp5^lstm_134/while/lstm_cell_134/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
lstm_134/while/Identity_2?
lstm_134/while/Identity_3IdentityClstm_134/while/TensorArrayV2Write/TensorListSetItem:output_handle:04^lstm_134/while/lstm_cell_134/BiasAdd/ReadVariableOp3^lstm_134/while/lstm_cell_134/MatMul/ReadVariableOp5^lstm_134/while/lstm_cell_134/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
lstm_134/while/Identity_3?
lstm_134/while/Identity_4Identity&lstm_134/while/lstm_cell_134/mul_2:z:04^lstm_134/while/lstm_cell_134/BiasAdd/ReadVariableOp3^lstm_134/while/lstm_cell_134/MatMul/ReadVariableOp5^lstm_134/while/lstm_cell_134/MatMul_1/ReadVariableOp*
T0*'
_output_shapes
:?????????22
lstm_134/while/Identity_4?
lstm_134/while/Identity_5Identity&lstm_134/while/lstm_cell_134/add_1:z:04^lstm_134/while/lstm_cell_134/BiasAdd/ReadVariableOp3^lstm_134/while/lstm_cell_134/MatMul/ReadVariableOp5^lstm_134/while/lstm_cell_134/MatMul_1/ReadVariableOp*
T0*'
_output_shapes
:?????????22
lstm_134/while/Identity_5";
lstm_134_while_identity lstm_134/while/Identity:output:0"?
lstm_134_while_identity_1"lstm_134/while/Identity_1:output:0"?
lstm_134_while_identity_2"lstm_134/while/Identity_2:output:0"?
lstm_134_while_identity_3"lstm_134/while/Identity_3:output:0"?
lstm_134_while_identity_4"lstm_134/while/Identity_4:output:0"?
lstm_134_while_identity_5"lstm_134/while/Identity_5:output:0"T
'lstm_134_while_lstm_134_strided_slice_1)lstm_134_while_lstm_134_strided_slice_1_0"~
<lstm_134_while_lstm_cell_134_biasadd_readvariableop_resource>lstm_134_while_lstm_cell_134_biasadd_readvariableop_resource_0"?
=lstm_134_while_lstm_cell_134_matmul_1_readvariableop_resource?lstm_134_while_lstm_cell_134_matmul_1_readvariableop_resource_0"|
;lstm_134_while_lstm_cell_134_matmul_readvariableop_resource=lstm_134_while_lstm_cell_134_matmul_readvariableop_resource_0"?
clstm_134_while_tensorarrayv2read_tensorlistgetitem_lstm_134_tensorarrayunstack_tensorlistfromtensorelstm_134_while_tensorarrayv2read_tensorlistgetitem_lstm_134_tensorarrayunstack_tensorlistfromtensor_0*Q
_input_shapes@
>: : : : :?????????2:?????????2: : :::2j
3lstm_134/while/lstm_cell_134/BiasAdd/ReadVariableOp3lstm_134/while/lstm_cell_134/BiasAdd/ReadVariableOp2h
2lstm_134/while/lstm_cell_134/MatMul/ReadVariableOp2lstm_134/while/lstm_cell_134/MatMul/ReadVariableOp2l
4lstm_134/while/lstm_cell_134/MatMul_1/ReadVariableOp4lstm_134/while/lstm_cell_134/MatMul_1/ReadVariableOp: 
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
:?????????2:-)
'
_output_shapes
:?????????2:

_output_shapes
: :

_output_shapes
: 
?C
?
while_body_1065235
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_08
4while_lstm_cell_134_matmul_readvariableop_resource_0:
6while_lstm_cell_134_matmul_1_readvariableop_resource_09
5while_lstm_cell_134_biasadd_readvariableop_resource_0
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor6
2while_lstm_cell_134_matmul_readvariableop_resource8
4while_lstm_cell_134_matmul_1_readvariableop_resource7
3while_lstm_cell_134_biasadd_readvariableop_resource??*while/lstm_cell_134/BiasAdd/ReadVariableOp?)while/lstm_cell_134/MatMul/ReadVariableOp?+while/lstm_cell_134/MatMul_1/ReadVariableOp?
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   29
7while/TensorArrayV2Read/TensorListGetItem/element_shape?
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:?????????*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItem?
)while/lstm_cell_134/MatMul/ReadVariableOpReadVariableOp4while_lstm_cell_134_matmul_readvariableop_resource_0*
_output_shapes
:	?*
dtype02+
)while/lstm_cell_134/MatMul/ReadVariableOp?
while/lstm_cell_134/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:01while/lstm_cell_134/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_134/MatMul?
+while/lstm_cell_134/MatMul_1/ReadVariableOpReadVariableOp6while_lstm_cell_134_matmul_1_readvariableop_resource_0*
_output_shapes
:	2?*
dtype02-
+while/lstm_cell_134/MatMul_1/ReadVariableOp?
while/lstm_cell_134/MatMul_1MatMulwhile_placeholder_23while/lstm_cell_134/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_134/MatMul_1?
while/lstm_cell_134/addAddV2$while/lstm_cell_134/MatMul:product:0&while/lstm_cell_134/MatMul_1:product:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_134/add?
*while/lstm_cell_134/BiasAdd/ReadVariableOpReadVariableOp5while_lstm_cell_134_biasadd_readvariableop_resource_0*
_output_shapes	
:?*
dtype02,
*while/lstm_cell_134/BiasAdd/ReadVariableOp?
while/lstm_cell_134/BiasAddBiasAddwhile/lstm_cell_134/add:z:02while/lstm_cell_134/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_134/BiasAddx
while/lstm_cell_134/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
while/lstm_cell_134/Const?
#while/lstm_cell_134/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2%
#while/lstm_cell_134/split/split_dim?
while/lstm_cell_134/splitSplit,while/lstm_cell_134/split/split_dim:output:0$while/lstm_cell_134/BiasAdd:output:0*
T0*`
_output_shapesN
L:?????????2:?????????2:?????????2:?????????2*
	num_split2
while/lstm_cell_134/split?
while/lstm_cell_134/SigmoidSigmoid"while/lstm_cell_134/split:output:0*
T0*'
_output_shapes
:?????????22
while/lstm_cell_134/Sigmoid?
while/lstm_cell_134/Sigmoid_1Sigmoid"while/lstm_cell_134/split:output:1*
T0*'
_output_shapes
:?????????22
while/lstm_cell_134/Sigmoid_1?
while/lstm_cell_134/mulMul!while/lstm_cell_134/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:?????????22
while/lstm_cell_134/mul?
while/lstm_cell_134/ReluRelu"while/lstm_cell_134/split:output:2*
T0*'
_output_shapes
:?????????22
while/lstm_cell_134/Relu?
while/lstm_cell_134/mul_1Mulwhile/lstm_cell_134/Sigmoid:y:0&while/lstm_cell_134/Relu:activations:0*
T0*'
_output_shapes
:?????????22
while/lstm_cell_134/mul_1?
while/lstm_cell_134/add_1AddV2while/lstm_cell_134/mul:z:0while/lstm_cell_134/mul_1:z:0*
T0*'
_output_shapes
:?????????22
while/lstm_cell_134/add_1?
while/lstm_cell_134/Sigmoid_2Sigmoid"while/lstm_cell_134/split:output:3*
T0*'
_output_shapes
:?????????22
while/lstm_cell_134/Sigmoid_2?
while/lstm_cell_134/Relu_1Reluwhile/lstm_cell_134/add_1:z:0*
T0*'
_output_shapes
:?????????22
while/lstm_cell_134/Relu_1?
while/lstm_cell_134/mul_2Mul!while/lstm_cell_134/Sigmoid_2:y:0(while/lstm_cell_134/Relu_1:activations:0*
T0*'
_output_shapes
:?????????22
while/lstm_cell_134/mul_2?
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_134/mul_2:z:0*
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
while/add_1?
while/IdentityIdentitywhile/add_1:z:0+^while/lstm_cell_134/BiasAdd/ReadVariableOp*^while/lstm_cell_134/MatMul/ReadVariableOp,^while/lstm_cell_134/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity?
while/Identity_1Identitywhile_while_maximum_iterations+^while/lstm_cell_134/BiasAdd/ReadVariableOp*^while/lstm_cell_134/MatMul/ReadVariableOp,^while/lstm_cell_134/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_1?
while/Identity_2Identitywhile/add:z:0+^while/lstm_cell_134/BiasAdd/ReadVariableOp*^while/lstm_cell_134/MatMul/ReadVariableOp,^while/lstm_cell_134/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_2?
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0+^while/lstm_cell_134/BiasAdd/ReadVariableOp*^while/lstm_cell_134/MatMul/ReadVariableOp,^while/lstm_cell_134/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_3?
while/Identity_4Identitywhile/lstm_cell_134/mul_2:z:0+^while/lstm_cell_134/BiasAdd/ReadVariableOp*^while/lstm_cell_134/MatMul/ReadVariableOp,^while/lstm_cell_134/MatMul_1/ReadVariableOp*
T0*'
_output_shapes
:?????????22
while/Identity_4?
while/Identity_5Identitywhile/lstm_cell_134/add_1:z:0+^while/lstm_cell_134/BiasAdd/ReadVariableOp*^while/lstm_cell_134/MatMul/ReadVariableOp,^while/lstm_cell_134/MatMul_1/ReadVariableOp*
T0*'
_output_shapes
:?????????22
while/Identity_5")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"l
3while_lstm_cell_134_biasadd_readvariableop_resource5while_lstm_cell_134_biasadd_readvariableop_resource_0"n
4while_lstm_cell_134_matmul_1_readvariableop_resource6while_lstm_cell_134_matmul_1_readvariableop_resource_0"j
2while_lstm_cell_134_matmul_readvariableop_resource4while_lstm_cell_134_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"?
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*Q
_input_shapes@
>: : : : :?????????2:?????????2: : :::2X
*while/lstm_cell_134/BiasAdd/ReadVariableOp*while/lstm_cell_134/BiasAdd/ReadVariableOp2V
)while/lstm_cell_134/MatMul/ReadVariableOp)while/lstm_cell_134/MatMul/ReadVariableOp2Z
+while/lstm_cell_134/MatMul_1/ReadVariableOp+while/lstm_cell_134/MatMul_1/ReadVariableOp: 
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
:?????????2:-)
'
_output_shapes
:?????????2:

_output_shapes
: :

_output_shapes
: 
?%
?
while_body_1064118
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0!
while_lstm_cell_134_1064142_0!
while_lstm_cell_134_1064144_0!
while_lstm_cell_134_1064146_0
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor
while_lstm_cell_134_1064142
while_lstm_cell_134_1064144
while_lstm_cell_134_1064146??+while/lstm_cell_134/StatefulPartitionedCall?
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   29
7while/TensorArrayV2Read/TensorListGetItem/element_shape?
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:?????????*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItem?
+while/lstm_cell_134/StatefulPartitionedCallStatefulPartitionedCall0while/TensorArrayV2Read/TensorListGetItem:item:0while_placeholder_2while_placeholder_3while_lstm_cell_134_1064142_0while_lstm_cell_134_1064144_0while_lstm_cell_134_1064146_0*
Tin

2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:?????????2:?????????2:?????????2*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *S
fNRL
J__inference_lstm_cell_134_layer_call_and_return_conditional_losses_10636922-
+while/lstm_cell_134/StatefulPartitionedCall?
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder4while/lstm_cell_134/StatefulPartitionedCall:output:0*
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
while/add_1?
while/IdentityIdentitywhile/add_1:z:0,^while/lstm_cell_134/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity?
while/Identity_1Identitywhile_while_maximum_iterations,^while/lstm_cell_134/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity_1?
while/Identity_2Identitywhile/add:z:0,^while/lstm_cell_134/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity_2?
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0,^while/lstm_cell_134/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity_3?
while/Identity_4Identity4while/lstm_cell_134/StatefulPartitionedCall:output:1,^while/lstm_cell_134/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????22
while/Identity_4?
while/Identity_5Identity4while/lstm_cell_134/StatefulPartitionedCall:output:2,^while/lstm_cell_134/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????22
while/Identity_5")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"<
while_lstm_cell_134_1064142while_lstm_cell_134_1064142_0"<
while_lstm_cell_134_1064144while_lstm_cell_134_1064144_0"<
while_lstm_cell_134_1064146while_lstm_cell_134_1064146_0"0
while_strided_slice_1while_strided_slice_1_0"?
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*Q
_input_shapes@
>: : : : :?????????2:?????????2: : :::2Z
+while/lstm_cell_134/StatefulPartitionedCall+while/lstm_cell_134/StatefulPartitionedCall: 
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
:?????????2:-)
'
_output_shapes
:?????????2:

_output_shapes
: :

_output_shapes
: 
?`
?
#__inference__traced_restore_1065954
file_prefix%
!assignvariableop_dense_128_kernel%
!assignvariableop_1_dense_128_bias 
assignvariableop_2_adam_iter"
assignvariableop_3_adam_beta_1"
assignvariableop_4_adam_beta_2!
assignvariableop_5_adam_decay)
%assignvariableop_6_adam_learning_rate4
0assignvariableop_7_lstm_134_lstm_cell_134_kernel>
:assignvariableop_8_lstm_134_lstm_cell_134_recurrent_kernel2
.assignvariableop_9_lstm_134_lstm_cell_134_bias
assignvariableop_10_total
assignvariableop_11_count/
+assignvariableop_12_adam_dense_128_kernel_m-
)assignvariableop_13_adam_dense_128_bias_m<
8assignvariableop_14_adam_lstm_134_lstm_cell_134_kernel_mF
Bassignvariableop_15_adam_lstm_134_lstm_cell_134_recurrent_kernel_m:
6assignvariableop_16_adam_lstm_134_lstm_cell_134_bias_m/
+assignvariableop_17_adam_dense_128_kernel_v-
)assignvariableop_18_adam_dense_128_bias_v<
8assignvariableop_19_adam_lstm_134_lstm_cell_134_kernel_vF
Bassignvariableop_20_adam_lstm_134_lstm_cell_134_recurrent_kernel_v:
6assignvariableop_21_adam_lstm_134_lstm_cell_134_bias_v
identity_23??AssignVariableOp?AssignVariableOp_1?AssignVariableOp_10?AssignVariableOp_11?AssignVariableOp_12?AssignVariableOp_13?AssignVariableOp_14?AssignVariableOp_15?AssignVariableOp_16?AssignVariableOp_17?AssignVariableOp_18?AssignVariableOp_19?AssignVariableOp_2?AssignVariableOp_20?AssignVariableOp_21?AssignVariableOp_3?AssignVariableOp_4?AssignVariableOp_5?AssignVariableOp_6?AssignVariableOp_7?AssignVariableOp_8?AssignVariableOp_9?
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*?

value?
B?
B6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2/tensor_names?
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*A
value8B6B B B B B B B B B B B B B B B B B B B B B B B 2
RestoreV2/shape_and_slices?
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

Identity?
AssignVariableOpAssignVariableOp!assignvariableop_dense_128_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1?
AssignVariableOp_1AssignVariableOp!assignvariableop_1_dense_128_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0	*
_output_shapes
:2

Identity_2?
AssignVariableOp_2AssignVariableOpassignvariableop_2_adam_iterIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	2
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3?
AssignVariableOp_3AssignVariableOpassignvariableop_3_adam_beta_1Identity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_3k

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:2

Identity_4?
AssignVariableOp_4AssignVariableOpassignvariableop_4_adam_beta_2Identity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_4k

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:2

Identity_5?
AssignVariableOp_5AssignVariableOpassignvariableop_5_adam_decayIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_5k

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:2

Identity_6?
AssignVariableOp_6AssignVariableOp%assignvariableop_6_adam_learning_rateIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_6k

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:2

Identity_7?
AssignVariableOp_7AssignVariableOp0assignvariableop_7_lstm_134_lstm_cell_134_kernelIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_7k

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:2

Identity_8?
AssignVariableOp_8AssignVariableOp:assignvariableop_8_lstm_134_lstm_cell_134_recurrent_kernelIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_8k

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:2

Identity_9?
AssignVariableOp_9AssignVariableOp.assignvariableop_9_lstm_134_lstm_cell_134_biasIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_9n
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:2
Identity_10?
AssignVariableOp_10AssignVariableOpassignvariableop_10_totalIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_10n
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:2
Identity_11?
AssignVariableOp_11AssignVariableOpassignvariableop_11_countIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_11n
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:2
Identity_12?
AssignVariableOp_12AssignVariableOp+assignvariableop_12_adam_dense_128_kernel_mIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_12n
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:2
Identity_13?
AssignVariableOp_13AssignVariableOp)assignvariableop_13_adam_dense_128_bias_mIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_13n
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:2
Identity_14?
AssignVariableOp_14AssignVariableOp8assignvariableop_14_adam_lstm_134_lstm_cell_134_kernel_mIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_14n
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:2
Identity_15?
AssignVariableOp_15AssignVariableOpBassignvariableop_15_adam_lstm_134_lstm_cell_134_recurrent_kernel_mIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_15n
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:2
Identity_16?
AssignVariableOp_16AssignVariableOp6assignvariableop_16_adam_lstm_134_lstm_cell_134_bias_mIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_16n
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:2
Identity_17?
AssignVariableOp_17AssignVariableOp+assignvariableop_17_adam_dense_128_kernel_vIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_17n
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:2
Identity_18?
AssignVariableOp_18AssignVariableOp)assignvariableop_18_adam_dense_128_bias_vIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_18n
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:2
Identity_19?
AssignVariableOp_19AssignVariableOp8assignvariableop_19_adam_lstm_134_lstm_cell_134_kernel_vIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_19n
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:2
Identity_20?
AssignVariableOp_20AssignVariableOpBassignvariableop_20_adam_lstm_134_lstm_cell_134_recurrent_kernel_vIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_20n
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:2
Identity_21?
AssignVariableOp_21AssignVariableOp6assignvariableop_21_adam_lstm_134_lstm_cell_134_bias_vIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_219
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOp?
Identity_22Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2
Identity_22?
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
?[
?
E__inference_lstm_134_layer_call_and_return_conditional_losses_1064352

inputs0
,lstm_cell_134_matmul_readvariableop_resource2
.lstm_cell_134_matmul_1_readvariableop_resource1
-lstm_cell_134_biasadd_readvariableop_resource
identity??$lstm_cell_134/BiasAdd/ReadVariableOp?#lstm_cell_134/MatMul/ReadVariableOp?%lstm_cell_134/MatMul_1/ReadVariableOp?whileD
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
strided_slice/stack_2?
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
value	B :22
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
B :?2
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
value	B :22
zeros/packed/1?
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
:?????????22
zeros`
zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B :22
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
B :?2
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
value	B :22
zeros_1/packed/1?
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
:?????????22	
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
:?????????2
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
strided_slice_1/stack_2?
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1?
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
TensorArrayV2/element_shape?
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2?
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   27
5TensorArrayUnstack/TensorListFromTensor/element_shape?
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
strided_slice_2/stack_2?
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*
shrink_axis_mask2
strided_slice_2?
#lstm_cell_134/MatMul/ReadVariableOpReadVariableOp,lstm_cell_134_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype02%
#lstm_cell_134/MatMul/ReadVariableOp?
lstm_cell_134/MatMulMatMulstrided_slice_2:output:0+lstm_cell_134/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
lstm_cell_134/MatMul?
%lstm_cell_134/MatMul_1/ReadVariableOpReadVariableOp.lstm_cell_134_matmul_1_readvariableop_resource*
_output_shapes
:	2?*
dtype02'
%lstm_cell_134/MatMul_1/ReadVariableOp?
lstm_cell_134/MatMul_1MatMulzeros:output:0-lstm_cell_134/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
lstm_cell_134/MatMul_1?
lstm_cell_134/addAddV2lstm_cell_134/MatMul:product:0 lstm_cell_134/MatMul_1:product:0*
T0*(
_output_shapes
:??????????2
lstm_cell_134/add?
$lstm_cell_134/BiasAdd/ReadVariableOpReadVariableOp-lstm_cell_134_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02&
$lstm_cell_134/BiasAdd/ReadVariableOp?
lstm_cell_134/BiasAddBiasAddlstm_cell_134/add:z:0,lstm_cell_134/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
lstm_cell_134/BiasAddl
lstm_cell_134/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_134/Const?
lstm_cell_134/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_134/split/split_dim?
lstm_cell_134/splitSplit&lstm_cell_134/split/split_dim:output:0lstm_cell_134/BiasAdd:output:0*
T0*`
_output_shapesN
L:?????????2:?????????2:?????????2:?????????2*
	num_split2
lstm_cell_134/split?
lstm_cell_134/SigmoidSigmoidlstm_cell_134/split:output:0*
T0*'
_output_shapes
:?????????22
lstm_cell_134/Sigmoid?
lstm_cell_134/Sigmoid_1Sigmoidlstm_cell_134/split:output:1*
T0*'
_output_shapes
:?????????22
lstm_cell_134/Sigmoid_1?
lstm_cell_134/mulMullstm_cell_134/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:?????????22
lstm_cell_134/mul?
lstm_cell_134/ReluRelulstm_cell_134/split:output:2*
T0*'
_output_shapes
:?????????22
lstm_cell_134/Relu?
lstm_cell_134/mul_1Mullstm_cell_134/Sigmoid:y:0 lstm_cell_134/Relu:activations:0*
T0*'
_output_shapes
:?????????22
lstm_cell_134/mul_1?
lstm_cell_134/add_1AddV2lstm_cell_134/mul:z:0lstm_cell_134/mul_1:z:0*
T0*'
_output_shapes
:?????????22
lstm_cell_134/add_1?
lstm_cell_134/Sigmoid_2Sigmoidlstm_cell_134/split:output:3*
T0*'
_output_shapes
:?????????22
lstm_cell_134/Sigmoid_2
lstm_cell_134/Relu_1Relulstm_cell_134/add_1:z:0*
T0*'
_output_shapes
:?????????22
lstm_cell_134/Relu_1?
lstm_cell_134/mul_2Mullstm_cell_134/Sigmoid_2:y:0"lstm_cell_134/Relu_1:activations:0*
T0*'
_output_shapes
:?????????22
lstm_cell_134/mul_2?
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????2   2
TensorArrayV2_1/element_shape?
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
?????????2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counter?
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0,lstm_cell_134_matmul_readvariableop_resource.lstm_cell_134_matmul_1_readvariableop_resource-lstm_cell_134_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :?????????2:?????????2: : : : : *%
_read_only_resource_inputs
	
*
bodyR
while_body_1064267*
condR
while_cond_1064266*K
output_shapes:
8: : : : :?????????2:?????????2: : : : : *
parallel_iterations 2
while?
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????2   22
0TensorArrayV2Stack/TensorListStack/element_shape?
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:?????????2*
element_dtype02$
"TensorArrayV2Stack/TensorListStack?
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????2
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
strided_slice_3/stack_2?
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????2*
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/perm?
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*+
_output_shapes
:?????????22
transpose_1f
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2	
runtime?
IdentityIdentitystrided_slice_3:output:0%^lstm_cell_134/BiasAdd/ReadVariableOp$^lstm_cell_134/MatMul/ReadVariableOp&^lstm_cell_134/MatMul_1/ReadVariableOp^while*
T0*'
_output_shapes
:?????????22

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????:::2L
$lstm_cell_134/BiasAdd/ReadVariableOp$lstm_cell_134/BiasAdd/ReadVariableOp2J
#lstm_cell_134/MatMul/ReadVariableOp#lstm_cell_134/MatMul/ReadVariableOp2N
%lstm_cell_134/MatMul_1/ReadVariableOp%lstm_cell_134/MatMul_1/ReadVariableOp2
whilewhile:S O
+
_output_shapes
:?????????
 
_user_specified_nameinputs
?C
?
while_body_1064267
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_08
4while_lstm_cell_134_matmul_readvariableop_resource_0:
6while_lstm_cell_134_matmul_1_readvariableop_resource_09
5while_lstm_cell_134_biasadd_readvariableop_resource_0
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor6
2while_lstm_cell_134_matmul_readvariableop_resource8
4while_lstm_cell_134_matmul_1_readvariableop_resource7
3while_lstm_cell_134_biasadd_readvariableop_resource??*while/lstm_cell_134/BiasAdd/ReadVariableOp?)while/lstm_cell_134/MatMul/ReadVariableOp?+while/lstm_cell_134/MatMul_1/ReadVariableOp?
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   29
7while/TensorArrayV2Read/TensorListGetItem/element_shape?
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:?????????*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItem?
)while/lstm_cell_134/MatMul/ReadVariableOpReadVariableOp4while_lstm_cell_134_matmul_readvariableop_resource_0*
_output_shapes
:	?*
dtype02+
)while/lstm_cell_134/MatMul/ReadVariableOp?
while/lstm_cell_134/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:01while/lstm_cell_134/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_134/MatMul?
+while/lstm_cell_134/MatMul_1/ReadVariableOpReadVariableOp6while_lstm_cell_134_matmul_1_readvariableop_resource_0*
_output_shapes
:	2?*
dtype02-
+while/lstm_cell_134/MatMul_1/ReadVariableOp?
while/lstm_cell_134/MatMul_1MatMulwhile_placeholder_23while/lstm_cell_134/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_134/MatMul_1?
while/lstm_cell_134/addAddV2$while/lstm_cell_134/MatMul:product:0&while/lstm_cell_134/MatMul_1:product:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_134/add?
*while/lstm_cell_134/BiasAdd/ReadVariableOpReadVariableOp5while_lstm_cell_134_biasadd_readvariableop_resource_0*
_output_shapes	
:?*
dtype02,
*while/lstm_cell_134/BiasAdd/ReadVariableOp?
while/lstm_cell_134/BiasAddBiasAddwhile/lstm_cell_134/add:z:02while/lstm_cell_134/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_134/BiasAddx
while/lstm_cell_134/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
while/lstm_cell_134/Const?
#while/lstm_cell_134/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2%
#while/lstm_cell_134/split/split_dim?
while/lstm_cell_134/splitSplit,while/lstm_cell_134/split/split_dim:output:0$while/lstm_cell_134/BiasAdd:output:0*
T0*`
_output_shapesN
L:?????????2:?????????2:?????????2:?????????2*
	num_split2
while/lstm_cell_134/split?
while/lstm_cell_134/SigmoidSigmoid"while/lstm_cell_134/split:output:0*
T0*'
_output_shapes
:?????????22
while/lstm_cell_134/Sigmoid?
while/lstm_cell_134/Sigmoid_1Sigmoid"while/lstm_cell_134/split:output:1*
T0*'
_output_shapes
:?????????22
while/lstm_cell_134/Sigmoid_1?
while/lstm_cell_134/mulMul!while/lstm_cell_134/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:?????????22
while/lstm_cell_134/mul?
while/lstm_cell_134/ReluRelu"while/lstm_cell_134/split:output:2*
T0*'
_output_shapes
:?????????22
while/lstm_cell_134/Relu?
while/lstm_cell_134/mul_1Mulwhile/lstm_cell_134/Sigmoid:y:0&while/lstm_cell_134/Relu:activations:0*
T0*'
_output_shapes
:?????????22
while/lstm_cell_134/mul_1?
while/lstm_cell_134/add_1AddV2while/lstm_cell_134/mul:z:0while/lstm_cell_134/mul_1:z:0*
T0*'
_output_shapes
:?????????22
while/lstm_cell_134/add_1?
while/lstm_cell_134/Sigmoid_2Sigmoid"while/lstm_cell_134/split:output:3*
T0*'
_output_shapes
:?????????22
while/lstm_cell_134/Sigmoid_2?
while/lstm_cell_134/Relu_1Reluwhile/lstm_cell_134/add_1:z:0*
T0*'
_output_shapes
:?????????22
while/lstm_cell_134/Relu_1?
while/lstm_cell_134/mul_2Mul!while/lstm_cell_134/Sigmoid_2:y:0(while/lstm_cell_134/Relu_1:activations:0*
T0*'
_output_shapes
:?????????22
while/lstm_cell_134/mul_2?
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_134/mul_2:z:0*
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
while/add_1?
while/IdentityIdentitywhile/add_1:z:0+^while/lstm_cell_134/BiasAdd/ReadVariableOp*^while/lstm_cell_134/MatMul/ReadVariableOp,^while/lstm_cell_134/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity?
while/Identity_1Identitywhile_while_maximum_iterations+^while/lstm_cell_134/BiasAdd/ReadVariableOp*^while/lstm_cell_134/MatMul/ReadVariableOp,^while/lstm_cell_134/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_1?
while/Identity_2Identitywhile/add:z:0+^while/lstm_cell_134/BiasAdd/ReadVariableOp*^while/lstm_cell_134/MatMul/ReadVariableOp,^while/lstm_cell_134/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_2?
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0+^while/lstm_cell_134/BiasAdd/ReadVariableOp*^while/lstm_cell_134/MatMul/ReadVariableOp,^while/lstm_cell_134/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_3?
while/Identity_4Identitywhile/lstm_cell_134/mul_2:z:0+^while/lstm_cell_134/BiasAdd/ReadVariableOp*^while/lstm_cell_134/MatMul/ReadVariableOp,^while/lstm_cell_134/MatMul_1/ReadVariableOp*
T0*'
_output_shapes
:?????????22
while/Identity_4?
while/Identity_5Identitywhile/lstm_cell_134/add_1:z:0+^while/lstm_cell_134/BiasAdd/ReadVariableOp*^while/lstm_cell_134/MatMul/ReadVariableOp,^while/lstm_cell_134/MatMul_1/ReadVariableOp*
T0*'
_output_shapes
:?????????22
while/Identity_5")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"l
3while_lstm_cell_134_biasadd_readvariableop_resource5while_lstm_cell_134_biasadd_readvariableop_resource_0"n
4while_lstm_cell_134_matmul_1_readvariableop_resource6while_lstm_cell_134_matmul_1_readvariableop_resource_0"j
2while_lstm_cell_134_matmul_readvariableop_resource4while_lstm_cell_134_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"?
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*Q
_input_shapes@
>: : : : :?????????2:?????????2: : :::2X
*while/lstm_cell_134/BiasAdd/ReadVariableOp*while/lstm_cell_134/BiasAdd/ReadVariableOp2V
)while/lstm_cell_134/MatMul/ReadVariableOp)while/lstm_cell_134/MatMul/ReadVariableOp2Z
+while/lstm_cell_134/MatMul_1/ReadVariableOp+while/lstm_cell_134/MatMul_1/ReadVariableOp: 
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
:?????????2:-)
'
_output_shapes
:?????????2:

_output_shapes
: :

_output_shapes
: 
?
?
K__inference_sequential_129_layer_call_and_return_conditional_losses_1064562
lstm_134_input
lstm_134_1064528
lstm_134_1064530
lstm_134_1064532
dense_128_1064556
dense_128_1064558
identity??!dense_128/StatefulPartitionedCall? lstm_134/StatefulPartitionedCall?
 lstm_134/StatefulPartitionedCallStatefulPartitionedCalllstm_134_inputlstm_134_1064528lstm_134_1064530lstm_134_1064532*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????2*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_lstm_134_layer_call_and_return_conditional_losses_10643522"
 lstm_134/StatefulPartitionedCall?
!dense_128/StatefulPartitionedCallStatefulPartitionedCall)lstm_134/StatefulPartitionedCall:output:0dense_128_1064556dense_128_1064558*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_dense_128_layer_call_and_return_conditional_losses_10645452#
!dense_128/StatefulPartitionedCall?
IdentityIdentity*dense_128/StatefulPartitionedCall:output:0"^dense_128/StatefulPartitionedCall!^lstm_134/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:?????????:::::2F
!dense_128/StatefulPartitionedCall!dense_128/StatefulPartitionedCall2D
 lstm_134/StatefulPartitionedCall lstm_134/StatefulPartitionedCall:[ W
+
_output_shapes
:?????????
(
_user_specified_namelstm_134_input
?\
?
E__inference_lstm_134_layer_call_and_return_conditional_losses_1065320
inputs_00
,lstm_cell_134_matmul_readvariableop_resource2
.lstm_cell_134_matmul_1_readvariableop_resource1
-lstm_cell_134_biasadd_readvariableop_resource
identity??$lstm_cell_134/BiasAdd/ReadVariableOp?#lstm_cell_134/MatMul/ReadVariableOp?%lstm_cell_134/MatMul_1/ReadVariableOp?whileF
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
strided_slice/stack_2?
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
value	B :22
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
B :?2
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
value	B :22
zeros/packed/1?
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
:?????????22
zeros`
zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B :22
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
B :?2
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
value	B :22
zeros_1/packed/1?
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
:?????????22	
zeros_1u
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/perm?
	transpose	Transposeinputs_0transpose/perm:output:0*
T0*4
_output_shapes"
 :??????????????????2
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
strided_slice_1/stack_2?
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1?
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
TensorArrayV2/element_shape?
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2?
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   27
5TensorArrayUnstack/TensorListFromTensor/element_shape?
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
strided_slice_2/stack_2?
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*
shrink_axis_mask2
strided_slice_2?
#lstm_cell_134/MatMul/ReadVariableOpReadVariableOp,lstm_cell_134_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype02%
#lstm_cell_134/MatMul/ReadVariableOp?
lstm_cell_134/MatMulMatMulstrided_slice_2:output:0+lstm_cell_134/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
lstm_cell_134/MatMul?
%lstm_cell_134/MatMul_1/ReadVariableOpReadVariableOp.lstm_cell_134_matmul_1_readvariableop_resource*
_output_shapes
:	2?*
dtype02'
%lstm_cell_134/MatMul_1/ReadVariableOp?
lstm_cell_134/MatMul_1MatMulzeros:output:0-lstm_cell_134/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
lstm_cell_134/MatMul_1?
lstm_cell_134/addAddV2lstm_cell_134/MatMul:product:0 lstm_cell_134/MatMul_1:product:0*
T0*(
_output_shapes
:??????????2
lstm_cell_134/add?
$lstm_cell_134/BiasAdd/ReadVariableOpReadVariableOp-lstm_cell_134_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02&
$lstm_cell_134/BiasAdd/ReadVariableOp?
lstm_cell_134/BiasAddBiasAddlstm_cell_134/add:z:0,lstm_cell_134/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
lstm_cell_134/BiasAddl
lstm_cell_134/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_134/Const?
lstm_cell_134/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_134/split/split_dim?
lstm_cell_134/splitSplit&lstm_cell_134/split/split_dim:output:0lstm_cell_134/BiasAdd:output:0*
T0*`
_output_shapesN
L:?????????2:?????????2:?????????2:?????????2*
	num_split2
lstm_cell_134/split?
lstm_cell_134/SigmoidSigmoidlstm_cell_134/split:output:0*
T0*'
_output_shapes
:?????????22
lstm_cell_134/Sigmoid?
lstm_cell_134/Sigmoid_1Sigmoidlstm_cell_134/split:output:1*
T0*'
_output_shapes
:?????????22
lstm_cell_134/Sigmoid_1?
lstm_cell_134/mulMullstm_cell_134/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:?????????22
lstm_cell_134/mul?
lstm_cell_134/ReluRelulstm_cell_134/split:output:2*
T0*'
_output_shapes
:?????????22
lstm_cell_134/Relu?
lstm_cell_134/mul_1Mullstm_cell_134/Sigmoid:y:0 lstm_cell_134/Relu:activations:0*
T0*'
_output_shapes
:?????????22
lstm_cell_134/mul_1?
lstm_cell_134/add_1AddV2lstm_cell_134/mul:z:0lstm_cell_134/mul_1:z:0*
T0*'
_output_shapes
:?????????22
lstm_cell_134/add_1?
lstm_cell_134/Sigmoid_2Sigmoidlstm_cell_134/split:output:3*
T0*'
_output_shapes
:?????????22
lstm_cell_134/Sigmoid_2
lstm_cell_134/Relu_1Relulstm_cell_134/add_1:z:0*
T0*'
_output_shapes
:?????????22
lstm_cell_134/Relu_1?
lstm_cell_134/mul_2Mullstm_cell_134/Sigmoid_2:y:0"lstm_cell_134/Relu_1:activations:0*
T0*'
_output_shapes
:?????????22
lstm_cell_134/mul_2?
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????2   2
TensorArrayV2_1/element_shape?
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
?????????2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counter?
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0,lstm_cell_134_matmul_readvariableop_resource.lstm_cell_134_matmul_1_readvariableop_resource-lstm_cell_134_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :?????????2:?????????2: : : : : *%
_read_only_resource_inputs
	
*
bodyR
while_body_1065235*
condR
while_cond_1065234*K
output_shapes:
8: : : : :?????????2:?????????2: : : : : *
parallel_iterations 2
while?
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????2   22
0TensorArrayV2Stack/TensorListStack/element_shape?
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :??????????????????2*
element_dtype02$
"TensorArrayV2Stack/TensorListStack?
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????2
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
strided_slice_3/stack_2?
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????2*
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/perm?
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*4
_output_shapes"
 :??????????????????22
transpose_1f
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2	
runtime?
IdentityIdentitystrided_slice_3:output:0%^lstm_cell_134/BiasAdd/ReadVariableOp$^lstm_cell_134/MatMul/ReadVariableOp&^lstm_cell_134/MatMul_1/ReadVariableOp^while*
T0*'
_output_shapes
:?????????22

Identity"
identityIdentity:output:0*?
_input_shapes.
,:??????????????????:::2L
$lstm_cell_134/BiasAdd/ReadVariableOp$lstm_cell_134/BiasAdd/ReadVariableOp2J
#lstm_cell_134/MatMul/ReadVariableOp#lstm_cell_134/MatMul/ReadVariableOp2N
%lstm_cell_134/MatMul_1/ReadVariableOp%lstm_cell_134/MatMul_1/ReadVariableOp2
whilewhile:^ Z
4
_output_shapes"
 :??????????????????
"
_user_specified_name
inputs/0
?
?
while_cond_1065234
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_15
1while_while_cond_1065234___redundant_placeholder05
1while_while_cond_1065234___redundant_placeholder15
1while_while_cond_1065234___redundant_placeholder25
1while_while_cond_1065234___redundant_placeholder3
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
@: : : : :?????????2:?????????2: ::::: 
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
:?????????2:-)
'
_output_shapes
:?????????2:

_output_shapes
: :

_output_shapes
:
?C
?
while_body_1065410
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_08
4while_lstm_cell_134_matmul_readvariableop_resource_0:
6while_lstm_cell_134_matmul_1_readvariableop_resource_09
5while_lstm_cell_134_biasadd_readvariableop_resource_0
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor6
2while_lstm_cell_134_matmul_readvariableop_resource8
4while_lstm_cell_134_matmul_1_readvariableop_resource7
3while_lstm_cell_134_biasadd_readvariableop_resource??*while/lstm_cell_134/BiasAdd/ReadVariableOp?)while/lstm_cell_134/MatMul/ReadVariableOp?+while/lstm_cell_134/MatMul_1/ReadVariableOp?
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   29
7while/TensorArrayV2Read/TensorListGetItem/element_shape?
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:?????????*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItem?
)while/lstm_cell_134/MatMul/ReadVariableOpReadVariableOp4while_lstm_cell_134_matmul_readvariableop_resource_0*
_output_shapes
:	?*
dtype02+
)while/lstm_cell_134/MatMul/ReadVariableOp?
while/lstm_cell_134/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:01while/lstm_cell_134/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_134/MatMul?
+while/lstm_cell_134/MatMul_1/ReadVariableOpReadVariableOp6while_lstm_cell_134_matmul_1_readvariableop_resource_0*
_output_shapes
:	2?*
dtype02-
+while/lstm_cell_134/MatMul_1/ReadVariableOp?
while/lstm_cell_134/MatMul_1MatMulwhile_placeholder_23while/lstm_cell_134/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_134/MatMul_1?
while/lstm_cell_134/addAddV2$while/lstm_cell_134/MatMul:product:0&while/lstm_cell_134/MatMul_1:product:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_134/add?
*while/lstm_cell_134/BiasAdd/ReadVariableOpReadVariableOp5while_lstm_cell_134_biasadd_readvariableop_resource_0*
_output_shapes	
:?*
dtype02,
*while/lstm_cell_134/BiasAdd/ReadVariableOp?
while/lstm_cell_134/BiasAddBiasAddwhile/lstm_cell_134/add:z:02while/lstm_cell_134/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_134/BiasAddx
while/lstm_cell_134/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
while/lstm_cell_134/Const?
#while/lstm_cell_134/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2%
#while/lstm_cell_134/split/split_dim?
while/lstm_cell_134/splitSplit,while/lstm_cell_134/split/split_dim:output:0$while/lstm_cell_134/BiasAdd:output:0*
T0*`
_output_shapesN
L:?????????2:?????????2:?????????2:?????????2*
	num_split2
while/lstm_cell_134/split?
while/lstm_cell_134/SigmoidSigmoid"while/lstm_cell_134/split:output:0*
T0*'
_output_shapes
:?????????22
while/lstm_cell_134/Sigmoid?
while/lstm_cell_134/Sigmoid_1Sigmoid"while/lstm_cell_134/split:output:1*
T0*'
_output_shapes
:?????????22
while/lstm_cell_134/Sigmoid_1?
while/lstm_cell_134/mulMul!while/lstm_cell_134/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:?????????22
while/lstm_cell_134/mul?
while/lstm_cell_134/ReluRelu"while/lstm_cell_134/split:output:2*
T0*'
_output_shapes
:?????????22
while/lstm_cell_134/Relu?
while/lstm_cell_134/mul_1Mulwhile/lstm_cell_134/Sigmoid:y:0&while/lstm_cell_134/Relu:activations:0*
T0*'
_output_shapes
:?????????22
while/lstm_cell_134/mul_1?
while/lstm_cell_134/add_1AddV2while/lstm_cell_134/mul:z:0while/lstm_cell_134/mul_1:z:0*
T0*'
_output_shapes
:?????????22
while/lstm_cell_134/add_1?
while/lstm_cell_134/Sigmoid_2Sigmoid"while/lstm_cell_134/split:output:3*
T0*'
_output_shapes
:?????????22
while/lstm_cell_134/Sigmoid_2?
while/lstm_cell_134/Relu_1Reluwhile/lstm_cell_134/add_1:z:0*
T0*'
_output_shapes
:?????????22
while/lstm_cell_134/Relu_1?
while/lstm_cell_134/mul_2Mul!while/lstm_cell_134/Sigmoid_2:y:0(while/lstm_cell_134/Relu_1:activations:0*
T0*'
_output_shapes
:?????????22
while/lstm_cell_134/mul_2?
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_134/mul_2:z:0*
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
while/add_1?
while/IdentityIdentitywhile/add_1:z:0+^while/lstm_cell_134/BiasAdd/ReadVariableOp*^while/lstm_cell_134/MatMul/ReadVariableOp,^while/lstm_cell_134/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity?
while/Identity_1Identitywhile_while_maximum_iterations+^while/lstm_cell_134/BiasAdd/ReadVariableOp*^while/lstm_cell_134/MatMul/ReadVariableOp,^while/lstm_cell_134/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_1?
while/Identity_2Identitywhile/add:z:0+^while/lstm_cell_134/BiasAdd/ReadVariableOp*^while/lstm_cell_134/MatMul/ReadVariableOp,^while/lstm_cell_134/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_2?
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0+^while/lstm_cell_134/BiasAdd/ReadVariableOp*^while/lstm_cell_134/MatMul/ReadVariableOp,^while/lstm_cell_134/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_3?
while/Identity_4Identitywhile/lstm_cell_134/mul_2:z:0+^while/lstm_cell_134/BiasAdd/ReadVariableOp*^while/lstm_cell_134/MatMul/ReadVariableOp,^while/lstm_cell_134/MatMul_1/ReadVariableOp*
T0*'
_output_shapes
:?????????22
while/Identity_4?
while/Identity_5Identitywhile/lstm_cell_134/add_1:z:0+^while/lstm_cell_134/BiasAdd/ReadVariableOp*^while/lstm_cell_134/MatMul/ReadVariableOp,^while/lstm_cell_134/MatMul_1/ReadVariableOp*
T0*'
_output_shapes
:?????????22
while/Identity_5")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"l
3while_lstm_cell_134_biasadd_readvariableop_resource5while_lstm_cell_134_biasadd_readvariableop_resource_0"n
4while_lstm_cell_134_matmul_1_readvariableop_resource6while_lstm_cell_134_matmul_1_readvariableop_resource_0"j
2while_lstm_cell_134_matmul_readvariableop_resource4while_lstm_cell_134_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"?
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*Q
_input_shapes@
>: : : : :?????????2:?????????2: : :::2X
*while/lstm_cell_134/BiasAdd/ReadVariableOp*while/lstm_cell_134/BiasAdd/ReadVariableOp2V
)while/lstm_cell_134/MatMul/ReadVariableOp)while/lstm_cell_134/MatMul/ReadVariableOp2Z
+while/lstm_cell_134/MatMul_1/ReadVariableOp+while/lstm_cell_134/MatMul_1/ReadVariableOp: 
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
:?????????2:-)
'
_output_shapes
:?????????2:

_output_shapes
: :

_output_shapes
: 
?
?
J__inference_lstm_cell_134_layer_call_and_return_conditional_losses_1063692

inputs

states
states_1"
matmul_readvariableop_resource$
 matmul_1_readvariableop_resource#
biasadd_readvariableop_resource
identity

identity_1

identity_2??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?MatMul_1/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	?*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
MatMul?
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource*
_output_shapes
:	2?*
dtype02
MatMul_1/ReadVariableOpz
MatMul_1MatMulstatesMatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2

MatMul_1l
addAddV2MatMul:product:0MatMul_1:product:0*
T0*(
_output_shapes
:??????????2
add?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOpy
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2	
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
split/split_dim?
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*`
_output_shapesN
L:?????????2:?????????2:?????????2:?????????2*
	num_split2
split_
SigmoidSigmoidsplit:output:0*
T0*'
_output_shapes
:?????????22	
Sigmoidc
	Sigmoid_1Sigmoidsplit:output:1*
T0*'
_output_shapes
:?????????22
	Sigmoid_1\
mulMulSigmoid_1:y:0states_1*
T0*'
_output_shapes
:?????????22
mulV
ReluRelusplit:output:2*
T0*'
_output_shapes
:?????????22
Reluh
mul_1MulSigmoid:y:0Relu:activations:0*
T0*'
_output_shapes
:?????????22
mul_1]
add_1AddV2mul:z:0	mul_1:z:0*
T0*'
_output_shapes
:?????????22
add_1c
	Sigmoid_2Sigmoidsplit:output:3*
T0*'
_output_shapes
:?????????22
	Sigmoid_2U
Relu_1Relu	add_1:z:0*
T0*'
_output_shapes
:?????????22
Relu_1l
mul_2MulSigmoid_2:y:0Relu_1:activations:0*
T0*'
_output_shapes
:?????????22
mul_2?
IdentityIdentity	mul_2:z:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*'
_output_shapes
:?????????22

Identity?

Identity_1Identity	mul_2:z:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*'
_output_shapes
:?????????22

Identity_1?

Identity_2Identity	add_1:z:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*'
_output_shapes
:?????????22

Identity_2"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*X
_input_shapesG
E:?????????:?????????2:?????????2:::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs:OK
'
_output_shapes
:?????????2
 
_user_specified_namestates:OK
'
_output_shapes
:?????????2
 
_user_specified_namestates
?	
?
F__inference_dense_128_layer_call_and_return_conditional_losses_1064545

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:2*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2	
BiasAdd?
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????2::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????2
 
_user_specified_nameinputs
?
?
%__inference_signature_wrapper_1064666
lstm_134_input
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCalllstm_134_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3*
Tin

2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*'
_read_only_resource_inputs	
*-
config_proto

CPU

GPU 2J 8? *+
f&R$
"__inference__wrapped_model_10635862
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:?????????:::::22
StatefulPartitionedCallStatefulPartitionedCall:[ W
+
_output_shapes
:?????????
(
_user_specified_namelstm_134_input
?
?
0__inference_sequential_129_layer_call_fn_1064610
lstm_134_input
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCalllstm_134_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3*
Tin

2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*'
_read_only_resource_inputs	
*-
config_proto

CPU

GPU 2J 8? *T
fORM
K__inference_sequential_129_layer_call_and_return_conditional_losses_10645972
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:?????????:::::22
StatefulPartitionedCallStatefulPartitionedCall:[ W
+
_output_shapes
:?????????
(
_user_specified_namelstm_134_input
?
?
while_cond_1065409
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_15
1while_while_cond_1065409___redundant_placeholder05
1while_while_cond_1065409___redundant_placeholder15
1while_while_cond_1065409___redundant_placeholder25
1while_while_cond_1065409___redundant_placeholder3
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
@: : : : :?????????2:?????????2: ::::: 
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
:?????????2:-)
'
_output_shapes
:?????????2:

_output_shapes
: :

_output_shapes
:
?
?
*__inference_lstm_134_layer_call_fn_1065670

inputs
unknown
	unknown_0
	unknown_1
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????2*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_lstm_134_layer_call_and_return_conditional_losses_10645052
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????22

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????:::22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
while_cond_1065562
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_15
1while_while_cond_1065562___redundant_placeholder05
1while_while_cond_1065562___redundant_placeholder15
1while_while_cond_1065562___redundant_placeholder25
1while_while_cond_1065562___redundant_placeholder3
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
@: : : : :?????????2:?????????2: ::::: 
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
:?????????2:-)
'
_output_shapes
:?????????2:

_output_shapes
: :

_output_shapes
:
?
?
while_cond_1064117
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_15
1while_while_cond_1064117___redundant_placeholder05
1while_while_cond_1064117___redundant_placeholder15
1while_while_cond_1064117___redundant_placeholder25
1while_while_cond_1064117___redundant_placeholder3
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
@: : : : :?????????2:?????????2: ::::: 
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
:?????????2:-)
'
_output_shapes
:?????????2:

_output_shapes
: :

_output_shapes
:
?D
?
E__inference_lstm_134_layer_call_and_return_conditional_losses_1064187

inputs
lstm_cell_134_1064105
lstm_cell_134_1064107
lstm_cell_134_1064109
identity??%lstm_cell_134/StatefulPartitionedCall?whileD
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
strided_slice/stack_2?
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
value	B :22
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
B :?2
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
value	B :22
zeros/packed/1?
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
:?????????22
zeros`
zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B :22
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
B :?2
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
value	B :22
zeros_1/packed/1?
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
:?????????22	
zeros_1u
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/perm?
	transpose	Transposeinputstranspose/perm:output:0*
T0*4
_output_shapes"
 :??????????????????2
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
strided_slice_1/stack_2?
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1?
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
TensorArrayV2/element_shape?
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2?
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   27
5TensorArrayUnstack/TensorListFromTensor/element_shape?
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
strided_slice_2/stack_2?
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*
shrink_axis_mask2
strided_slice_2?
%lstm_cell_134/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_2:output:0zeros:output:0zeros_1:output:0lstm_cell_134_1064105lstm_cell_134_1064107lstm_cell_134_1064109*
Tin

2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:?????????2:?????????2:?????????2*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *S
fNRL
J__inference_lstm_cell_134_layer_call_and_return_conditional_losses_10636922'
%lstm_cell_134/StatefulPartitionedCall?
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????2   2
TensorArrayV2_1/element_shape?
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
?????????2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counter?
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0lstm_cell_134_1064105lstm_cell_134_1064107lstm_cell_134_1064109*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :?????????2:?????????2: : : : : *%
_read_only_resource_inputs
	
*
bodyR
while_body_1064118*
condR
while_cond_1064117*K
output_shapes:
8: : : : :?????????2:?????????2: : : : : *
parallel_iterations 2
while?
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????2   22
0TensorArrayV2Stack/TensorListStack/element_shape?
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :??????????????????2*
element_dtype02$
"TensorArrayV2Stack/TensorListStack?
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????2
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
strided_slice_3/stack_2?
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????2*
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/perm?
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*4
_output_shapes"
 :??????????????????22
transpose_1f
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2	
runtime?
IdentityIdentitystrided_slice_3:output:0&^lstm_cell_134/StatefulPartitionedCall^while*
T0*'
_output_shapes
:?????????22

Identity"
identityIdentity:output:0*?
_input_shapes.
,:??????????????????:::2N
%lstm_cell_134/StatefulPartitionedCall%lstm_cell_134/StatefulPartitionedCall2
whilewhile:\ X
4
_output_shapes"
 :??????????????????
 
_user_specified_nameinputs
?
?
while_cond_1064419
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_15
1while_while_cond_1064419___redundant_placeholder05
1while_while_cond_1064419___redundant_placeholder15
1while_while_cond_1064419___redundant_placeholder25
1while_while_cond_1064419___redundant_placeholder3
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
@: : : : :?????????2:?????????2: ::::: 
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
:?????????2:-)
'
_output_shapes
:?????????2:

_output_shapes
: :

_output_shapes
:
?
?
*__inference_lstm_134_layer_call_fn_1065331
inputs_0
unknown
	unknown_0
	unknown_1
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputs_0unknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????2*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_lstm_134_layer_call_and_return_conditional_losses_10640552
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????22

Identity"
identityIdentity:output:0*?
_input_shapes.
,:??????????????????:::22
StatefulPartitionedCallStatefulPartitionedCall:^ Z
4
_output_shapes"
 :??????????????????
"
_user_specified_name
inputs/0
?	
?
F__inference_dense_128_layer_call_and_return_conditional_losses_1065680

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:2*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2	
BiasAdd?
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????2::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????2
 
_user_specified_nameinputs
?u
?
K__inference_sequential_129_layer_call_and_return_conditional_losses_1064984

inputs9
5lstm_134_lstm_cell_134_matmul_readvariableop_resource;
7lstm_134_lstm_cell_134_matmul_1_readvariableop_resource:
6lstm_134_lstm_cell_134_biasadd_readvariableop_resource,
(dense_128_matmul_readvariableop_resource-
)dense_128_biasadd_readvariableop_resource
identity?? dense_128/BiasAdd/ReadVariableOp?dense_128/MatMul/ReadVariableOp?-lstm_134/lstm_cell_134/BiasAdd/ReadVariableOp?,lstm_134/lstm_cell_134/MatMul/ReadVariableOp?.lstm_134/lstm_cell_134/MatMul_1/ReadVariableOp?lstm_134/whileV
lstm_134/ShapeShapeinputs*
T0*
_output_shapes
:2
lstm_134/Shape?
lstm_134/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
lstm_134/strided_slice/stack?
lstm_134/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2 
lstm_134/strided_slice/stack_1?
lstm_134/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2 
lstm_134/strided_slice/stack_2?
lstm_134/strided_sliceStridedSlicelstm_134/Shape:output:0%lstm_134/strided_slice/stack:output:0'lstm_134/strided_slice/stack_1:output:0'lstm_134/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
lstm_134/strided_slicen
lstm_134/zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :22
lstm_134/zeros/mul/y?
lstm_134/zeros/mulMullstm_134/strided_slice:output:0lstm_134/zeros/mul/y:output:0*
T0*
_output_shapes
: 2
lstm_134/zeros/mulq
lstm_134/zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2
lstm_134/zeros/Less/y?
lstm_134/zeros/LessLesslstm_134/zeros/mul:z:0lstm_134/zeros/Less/y:output:0*
T0*
_output_shapes
: 2
lstm_134/zeros/Lesst
lstm_134/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :22
lstm_134/zeros/packed/1?
lstm_134/zeros/packedPacklstm_134/strided_slice:output:0 lstm_134/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
lstm_134/zeros/packedq
lstm_134/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
lstm_134/zeros/Const?
lstm_134/zerosFilllstm_134/zeros/packed:output:0lstm_134/zeros/Const:output:0*
T0*'
_output_shapes
:?????????22
lstm_134/zerosr
lstm_134/zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B :22
lstm_134/zeros_1/mul/y?
lstm_134/zeros_1/mulMullstm_134/strided_slice:output:0lstm_134/zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
lstm_134/zeros_1/mulu
lstm_134/zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2
lstm_134/zeros_1/Less/y?
lstm_134/zeros_1/LessLesslstm_134/zeros_1/mul:z:0 lstm_134/zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
lstm_134/zeros_1/Lessx
lstm_134/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :22
lstm_134/zeros_1/packed/1?
lstm_134/zeros_1/packedPacklstm_134/strided_slice:output:0"lstm_134/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
lstm_134/zeros_1/packedu
lstm_134/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
lstm_134/zeros_1/Const?
lstm_134/zeros_1Fill lstm_134/zeros_1/packed:output:0lstm_134/zeros_1/Const:output:0*
T0*'
_output_shapes
:?????????22
lstm_134/zeros_1?
lstm_134/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
lstm_134/transpose/perm?
lstm_134/transpose	Transposeinputs lstm_134/transpose/perm:output:0*
T0*+
_output_shapes
:?????????2
lstm_134/transposej
lstm_134/Shape_1Shapelstm_134/transpose:y:0*
T0*
_output_shapes
:2
lstm_134/Shape_1?
lstm_134/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2 
lstm_134/strided_slice_1/stack?
 lstm_134/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2"
 lstm_134/strided_slice_1/stack_1?
 lstm_134/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2"
 lstm_134/strided_slice_1/stack_2?
lstm_134/strided_slice_1StridedSlicelstm_134/Shape_1:output:0'lstm_134/strided_slice_1/stack:output:0)lstm_134/strided_slice_1/stack_1:output:0)lstm_134/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
lstm_134/strided_slice_1?
$lstm_134/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
?????????2&
$lstm_134/TensorArrayV2/element_shape?
lstm_134/TensorArrayV2TensorListReserve-lstm_134/TensorArrayV2/element_shape:output:0!lstm_134/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
lstm_134/TensorArrayV2?
>lstm_134/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   2@
>lstm_134/TensorArrayUnstack/TensorListFromTensor/element_shape?
0lstm_134/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorlstm_134/transpose:y:0Glstm_134/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type022
0lstm_134/TensorArrayUnstack/TensorListFromTensor?
lstm_134/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2 
lstm_134/strided_slice_2/stack?
 lstm_134/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2"
 lstm_134/strided_slice_2/stack_1?
 lstm_134/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2"
 lstm_134/strided_slice_2/stack_2?
lstm_134/strided_slice_2StridedSlicelstm_134/transpose:y:0'lstm_134/strided_slice_2/stack:output:0)lstm_134/strided_slice_2/stack_1:output:0)lstm_134/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*
shrink_axis_mask2
lstm_134/strided_slice_2?
,lstm_134/lstm_cell_134/MatMul/ReadVariableOpReadVariableOp5lstm_134_lstm_cell_134_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype02.
,lstm_134/lstm_cell_134/MatMul/ReadVariableOp?
lstm_134/lstm_cell_134/MatMulMatMul!lstm_134/strided_slice_2:output:04lstm_134/lstm_cell_134/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
lstm_134/lstm_cell_134/MatMul?
.lstm_134/lstm_cell_134/MatMul_1/ReadVariableOpReadVariableOp7lstm_134_lstm_cell_134_matmul_1_readvariableop_resource*
_output_shapes
:	2?*
dtype020
.lstm_134/lstm_cell_134/MatMul_1/ReadVariableOp?
lstm_134/lstm_cell_134/MatMul_1MatMullstm_134/zeros:output:06lstm_134/lstm_cell_134/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2!
lstm_134/lstm_cell_134/MatMul_1?
lstm_134/lstm_cell_134/addAddV2'lstm_134/lstm_cell_134/MatMul:product:0)lstm_134/lstm_cell_134/MatMul_1:product:0*
T0*(
_output_shapes
:??????????2
lstm_134/lstm_cell_134/add?
-lstm_134/lstm_cell_134/BiasAdd/ReadVariableOpReadVariableOp6lstm_134_lstm_cell_134_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02/
-lstm_134/lstm_cell_134/BiasAdd/ReadVariableOp?
lstm_134/lstm_cell_134/BiasAddBiasAddlstm_134/lstm_cell_134/add:z:05lstm_134/lstm_cell_134/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2 
lstm_134/lstm_cell_134/BiasAdd~
lstm_134/lstm_cell_134/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_134/lstm_cell_134/Const?
&lstm_134/lstm_cell_134/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2(
&lstm_134/lstm_cell_134/split/split_dim?
lstm_134/lstm_cell_134/splitSplit/lstm_134/lstm_cell_134/split/split_dim:output:0'lstm_134/lstm_cell_134/BiasAdd:output:0*
T0*`
_output_shapesN
L:?????????2:?????????2:?????????2:?????????2*
	num_split2
lstm_134/lstm_cell_134/split?
lstm_134/lstm_cell_134/SigmoidSigmoid%lstm_134/lstm_cell_134/split:output:0*
T0*'
_output_shapes
:?????????22 
lstm_134/lstm_cell_134/Sigmoid?
 lstm_134/lstm_cell_134/Sigmoid_1Sigmoid%lstm_134/lstm_cell_134/split:output:1*
T0*'
_output_shapes
:?????????22"
 lstm_134/lstm_cell_134/Sigmoid_1?
lstm_134/lstm_cell_134/mulMul$lstm_134/lstm_cell_134/Sigmoid_1:y:0lstm_134/zeros_1:output:0*
T0*'
_output_shapes
:?????????22
lstm_134/lstm_cell_134/mul?
lstm_134/lstm_cell_134/ReluRelu%lstm_134/lstm_cell_134/split:output:2*
T0*'
_output_shapes
:?????????22
lstm_134/lstm_cell_134/Relu?
lstm_134/lstm_cell_134/mul_1Mul"lstm_134/lstm_cell_134/Sigmoid:y:0)lstm_134/lstm_cell_134/Relu:activations:0*
T0*'
_output_shapes
:?????????22
lstm_134/lstm_cell_134/mul_1?
lstm_134/lstm_cell_134/add_1AddV2lstm_134/lstm_cell_134/mul:z:0 lstm_134/lstm_cell_134/mul_1:z:0*
T0*'
_output_shapes
:?????????22
lstm_134/lstm_cell_134/add_1?
 lstm_134/lstm_cell_134/Sigmoid_2Sigmoid%lstm_134/lstm_cell_134/split:output:3*
T0*'
_output_shapes
:?????????22"
 lstm_134/lstm_cell_134/Sigmoid_2?
lstm_134/lstm_cell_134/Relu_1Relu lstm_134/lstm_cell_134/add_1:z:0*
T0*'
_output_shapes
:?????????22
lstm_134/lstm_cell_134/Relu_1?
lstm_134/lstm_cell_134/mul_2Mul$lstm_134/lstm_cell_134/Sigmoid_2:y:0+lstm_134/lstm_cell_134/Relu_1:activations:0*
T0*'
_output_shapes
:?????????22
lstm_134/lstm_cell_134/mul_2?
&lstm_134/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????2   2(
&lstm_134/TensorArrayV2_1/element_shape?
lstm_134/TensorArrayV2_1TensorListReserve/lstm_134/TensorArrayV2_1/element_shape:output:0!lstm_134/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
lstm_134/TensorArrayV2_1`
lstm_134/timeConst*
_output_shapes
: *
dtype0*
value	B : 2
lstm_134/time?
!lstm_134/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????2#
!lstm_134/while/maximum_iterations|
lstm_134/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
lstm_134/while/loop_counter?
lstm_134/whileWhile$lstm_134/while/loop_counter:output:0*lstm_134/while/maximum_iterations:output:0lstm_134/time:output:0!lstm_134/TensorArrayV2_1:handle:0lstm_134/zeros:output:0lstm_134/zeros_1:output:0!lstm_134/strided_slice_1:output:0@lstm_134/TensorArrayUnstack/TensorListFromTensor:output_handle:05lstm_134_lstm_cell_134_matmul_readvariableop_resource7lstm_134_lstm_cell_134_matmul_1_readvariableop_resource6lstm_134_lstm_cell_134_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :?????????2:?????????2: : : : : *%
_read_only_resource_inputs
	
*'
bodyR
lstm_134_while_body_1064893*'
condR
lstm_134_while_cond_1064892*K
output_shapes:
8: : : : :?????????2:?????????2: : : : : *
parallel_iterations 2
lstm_134/while?
9lstm_134/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????2   2;
9lstm_134/TensorArrayV2Stack/TensorListStack/element_shape?
+lstm_134/TensorArrayV2Stack/TensorListStackTensorListStacklstm_134/while:output:3Blstm_134/TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:?????????2*
element_dtype02-
+lstm_134/TensorArrayV2Stack/TensorListStack?
lstm_134/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????2 
lstm_134/strided_slice_3/stack?
 lstm_134/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2"
 lstm_134/strided_slice_3/stack_1?
 lstm_134/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2"
 lstm_134/strided_slice_3/stack_2?
lstm_134/strided_slice_3StridedSlice4lstm_134/TensorArrayV2Stack/TensorListStack:tensor:0'lstm_134/strided_slice_3/stack:output:0)lstm_134/strided_slice_3/stack_1:output:0)lstm_134/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????2*
shrink_axis_mask2
lstm_134/strided_slice_3?
lstm_134/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
lstm_134/transpose_1/perm?
lstm_134/transpose_1	Transpose4lstm_134/TensorArrayV2Stack/TensorListStack:tensor:0"lstm_134/transpose_1/perm:output:0*
T0*+
_output_shapes
:?????????22
lstm_134/transpose_1x
lstm_134/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2
lstm_134/runtime?
dense_128/MatMul/ReadVariableOpReadVariableOp(dense_128_matmul_readvariableop_resource*
_output_shapes

:2*
dtype02!
dense_128/MatMul/ReadVariableOp?
dense_128/MatMulMatMul!lstm_134/strided_slice_3:output:0'dense_128/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_128/MatMul?
 dense_128/BiasAdd/ReadVariableOpReadVariableOp)dense_128_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02"
 dense_128/BiasAdd/ReadVariableOp?
dense_128/BiasAddBiasAdddense_128/MatMul:product:0(dense_128/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_128/BiasAdd?
IdentityIdentitydense_128/BiasAdd:output:0!^dense_128/BiasAdd/ReadVariableOp ^dense_128/MatMul/ReadVariableOp.^lstm_134/lstm_cell_134/BiasAdd/ReadVariableOp-^lstm_134/lstm_cell_134/MatMul/ReadVariableOp/^lstm_134/lstm_cell_134/MatMul_1/ReadVariableOp^lstm_134/while*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:?????????:::::2D
 dense_128/BiasAdd/ReadVariableOp dense_128/BiasAdd/ReadVariableOp2B
dense_128/MatMul/ReadVariableOpdense_128/MatMul/ReadVariableOp2^
-lstm_134/lstm_cell_134/BiasAdd/ReadVariableOp-lstm_134/lstm_cell_134/BiasAdd/ReadVariableOp2\
,lstm_134/lstm_cell_134/MatMul/ReadVariableOp,lstm_134/lstm_cell_134/MatMul/ReadVariableOp2`
.lstm_134/lstm_cell_134/MatMul_1/ReadVariableOp.lstm_134/lstm_cell_134/MatMul_1/ReadVariableOp2 
lstm_134/whilelstm_134/while:S O
+
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
0__inference_sequential_129_layer_call_fn_1064999

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3*
Tin

2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*'
_read_only_resource_inputs	
*-
config_proto

CPU

GPU 2J 8? *T
fORM
K__inference_sequential_129_layer_call_and_return_conditional_losses_10645972
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:?????????:::::22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:?????????
 
_user_specified_nameinputs"?L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*?
serving_default?
M
lstm_134_input;
 serving_default_lstm_134_input:0?????????=
	dense_1280
StatefulPartitionedCall:0?????????tensorflow/serving/predict:??
?"
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
G__call__"? 
_tf_keras_sequential?{"class_name": "Sequential", "name": "sequential_129", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "config": {"name": "sequential_129", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 20, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "lstm_134_input"}}, {"class_name": "LSTM", "config": {"name": "lstm_134", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 20, 1]}, "dtype": "float32", "return_sequences": false, "return_state": false, "go_backwards": false, "stateful": false, "unroll": false, "time_major": false, "units": 50, "activation": "relu", "recurrent_activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "recurrent_initializer": {"class_name": "Orthogonal", "config": {"gain": 1.0, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "unit_forget_bias": true, "kernel_regularizer": null, "recurrent_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "recurrent_constraint": null, "bias_constraint": null, "dropout": 0.0, "recurrent_dropout": 0.0, "implementation": 2}}, {"class_name": "Dense", "config": {"name": "dense_128", "trainable": true, "dtype": "float32", "units": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}]}, "input_spec": [{"class_name": "InputSpec", "config": {"dtype": null, "shape": {"class_name": "__tuple__", "items": [null, null, 1]}, "ndim": 3, "max_ndim": null, "min_ndim": null, "axes": {}}}], "build_input_shape": {"class_name": "TensorShape", "items": [null, 20, 1]}, "is_graph_network": true, "keras_version": "2.4.0", "backend": "tensorflow", "model_config": {"class_name": "Sequential", "config": {"name": "sequential_129", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 20, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "lstm_134_input"}}, {"class_name": "LSTM", "config": {"name": "lstm_134", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 20, 1]}, "dtype": "float32", "return_sequences": false, "return_state": false, "go_backwards": false, "stateful": false, "unroll": false, "time_major": false, "units": 50, "activation": "relu", "recurrent_activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "recurrent_initializer": {"class_name": "Orthogonal", "config": {"gain": 1.0, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "unit_forget_bias": true, "kernel_regularizer": null, "recurrent_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "recurrent_constraint": null, "bias_constraint": null, "dropout": 0.0, "recurrent_dropout": 0.0, "implementation": 2}}, {"class_name": "Dense", "config": {"name": "dense_128", "trainable": true, "dtype": "float32", "units": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}]}}, "training_config": {"loss": "mean_squared_error", "metrics": null, "weighted_metrics": null, "loss_weights": null, "optimizer_config": {"class_name": "Adam", "config": {"name": "Adam", "learning_rate": 0.0010000000474974513, "decay": 0.0, "beta_1": 0.8999999761581421, "beta_2": 0.9990000128746033, "epsilon": 1e-07, "amsgrad": false}}}}
?
	cell


state_spec
	variables
regularization_losses
trainable_variables
	keras_api
*H&call_and_return_all_conditional_losses
I__call__"?
_tf_keras_rnn_layer?
{"class_name": "LSTM", "name": "lstm_134", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": {"class_name": "__tuple__", "items": [null, 20, 1]}, "stateful": false, "must_restore_from_config": false, "config": {"name": "lstm_134", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 20, 1]}, "dtype": "float32", "return_sequences": false, "return_state": false, "go_backwards": false, "stateful": false, "unroll": false, "time_major": false, "units": 50, "activation": "relu", "recurrent_activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "recurrent_initializer": {"class_name": "Orthogonal", "config": {"gain": 1.0, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "unit_forget_bias": true, "kernel_regularizer": null, "recurrent_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "recurrent_constraint": null, "bias_constraint": null, "dropout": 0.0, "recurrent_dropout": 0.0, "implementation": 2}, "input_spec": [{"class_name": "InputSpec", "config": {"dtype": null, "shape": {"class_name": "__tuple__", "items": [null, null, 1]}, "ndim": 3, "max_ndim": null, "min_ndim": null, "axes": {}}}], "build_input_shape": {"class_name": "TensorShape", "items": [null, 20, 1]}}
?

kernel
bias
	variables
regularization_losses
trainable_variables
	keras_api
*J&call_and_return_all_conditional_losses
K__call__"?
_tf_keras_layer?{"class_name": "Dense", "name": "dense_128", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_128", "trainable": true, "dtype": "float32", "units": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 50}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 50]}}
?
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
?
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
?

kernel
recurrent_kernel
bias
"	variables
#regularization_losses
$trainable_variables
%	keras_api
*M&call_and_return_all_conditional_losses
N__call__"?
_tf_keras_layer?{"class_name": "LSTMCell", "name": "lstm_cell_134", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "lstm_cell_134", "trainable": true, "dtype": "float32", "units": 50, "activation": "relu", "recurrent_activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "recurrent_initializer": {"class_name": "Orthogonal", "config": {"gain": 1.0, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "unit_forget_bias": true, "kernel_regularizer": null, "recurrent_regularizer": null, "bias_regularizer": null, "kernel_constraint": null, "recurrent_constraint": null, "bias_constraint": null, "dropout": 0.0, "recurrent_dropout": 0.0, "implementation": 2}}
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
?
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
": 22dense_128/kernel
:2dense_128/bias
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
?
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
0:.	?2lstm_134/lstm_cell_134/kernel
::8	2?2'lstm_134/lstm_cell_134/recurrent_kernel
*:(?2lstm_134/lstm_cell_134/bias
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
?
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
?
	7total
	8count
9	variables
:	keras_api"?
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
':%22Adam/dense_128/kernel/m
!:2Adam/dense_128/bias/m
5:3	?2$Adam/lstm_134/lstm_cell_134/kernel/m
?:=	2?2.Adam/lstm_134/lstm_cell_134/recurrent_kernel/m
/:-?2"Adam/lstm_134/lstm_cell_134/bias/m
':%22Adam/dense_128/kernel/v
!:2Adam/dense_128/bias/v
5:3	?2$Adam/lstm_134/lstm_cell_134/kernel/v
?:=	2?2.Adam/lstm_134/lstm_cell_134/recurrent_kernel/v
/:-?2"Adam/lstm_134/lstm_cell_134/bias/v
?2?
"__inference__wrapped_model_1063586?
???
FullArgSpec
args? 
varargsjargs
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *1?.
,?)
lstm_134_input?????????
?2?
K__inference_sequential_129_layer_call_and_return_conditional_losses_1064578
K__inference_sequential_129_layer_call_and_return_conditional_losses_1064562
K__inference_sequential_129_layer_call_and_return_conditional_losses_1064825
K__inference_sequential_129_layer_call_and_return_conditional_losses_1064984?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
0__inference_sequential_129_layer_call_fn_1064610
0__inference_sequential_129_layer_call_fn_1065014
0__inference_sequential_129_layer_call_fn_1064641
0__inference_sequential_129_layer_call_fn_1064999?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
E__inference_lstm_134_layer_call_and_return_conditional_losses_1065167
E__inference_lstm_134_layer_call_and_return_conditional_losses_1065495
E__inference_lstm_134_layer_call_and_return_conditional_losses_1065320
E__inference_lstm_134_layer_call_and_return_conditional_losses_1065648?
???
FullArgSpecB
args:?7
jself
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaults?

 
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
*__inference_lstm_134_layer_call_fn_1065670
*__inference_lstm_134_layer_call_fn_1065342
*__inference_lstm_134_layer_call_fn_1065331
*__inference_lstm_134_layer_call_fn_1065659?
???
FullArgSpecB
args:?7
jself
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaults?

 
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
F__inference_dense_128_layer_call_and_return_conditional_losses_1065680?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
+__inference_dense_128_layer_call_fn_1065689?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?B?
%__inference_signature_wrapper_1064666lstm_134_input"?
???
FullArgSpec
args? 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
J__inference_lstm_cell_134_layer_call_and_return_conditional_losses_1065755
J__inference_lstm_cell_134_layer_call_and_return_conditional_losses_1065722?
???
FullArgSpec3
args+?(
jself
jinputs
jstates

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
/__inference_lstm_cell_134_layer_call_fn_1065772
/__inference_lstm_cell_134_layer_call_fn_1065789?
???
FullArgSpec3
args+?(
jself
jinputs
jstates

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 ?
"__inference__wrapped_model_1063586{;?8
1?.
,?)
lstm_134_input?????????
? "5?2
0
	dense_128#? 
	dense_128??????????
F__inference_dense_128_layer_call_and_return_conditional_losses_1065680\/?,
%?"
 ?
inputs?????????2
? "%?"
?
0?????????
? ~
+__inference_dense_128_layer_call_fn_1065689O/?,
%?"
 ?
inputs?????????2
? "???????????
E__inference_lstm_134_layer_call_and_return_conditional_losses_1065167}O?L
E?B
4?1
/?,
inputs/0??????????????????

 
p

 
? "%?"
?
0?????????2
? ?
E__inference_lstm_134_layer_call_and_return_conditional_losses_1065320}O?L
E?B
4?1
/?,
inputs/0??????????????????

 
p 

 
? "%?"
?
0?????????2
? ?
E__inference_lstm_134_layer_call_and_return_conditional_losses_1065495m??<
5?2
$?!
inputs?????????

 
p

 
? "%?"
?
0?????????2
? ?
E__inference_lstm_134_layer_call_and_return_conditional_losses_1065648m??<
5?2
$?!
inputs?????????

 
p 

 
? "%?"
?
0?????????2
? ?
*__inference_lstm_134_layer_call_fn_1065331pO?L
E?B
4?1
/?,
inputs/0??????????????????

 
p

 
? "??????????2?
*__inference_lstm_134_layer_call_fn_1065342pO?L
E?B
4?1
/?,
inputs/0??????????????????

 
p 

 
? "??????????2?
*__inference_lstm_134_layer_call_fn_1065659`??<
5?2
$?!
inputs?????????

 
p

 
? "??????????2?
*__inference_lstm_134_layer_call_fn_1065670`??<
5?2
$?!
inputs?????????

 
p 

 
? "??????????2?
J__inference_lstm_cell_134_layer_call_and_return_conditional_losses_1065722???}
v?s
 ?
inputs?????????
K?H
"?
states/0?????????2
"?
states/1?????????2
p
? "s?p
i?f
?
0/0?????????2
E?B
?
0/1/0?????????2
?
0/1/1?????????2
? ?
J__inference_lstm_cell_134_layer_call_and_return_conditional_losses_1065755???}
v?s
 ?
inputs?????????
K?H
"?
states/0?????????2
"?
states/1?????????2
p 
? "s?p
i?f
?
0/0?????????2
E?B
?
0/1/0?????????2
?
0/1/1?????????2
? ?
/__inference_lstm_cell_134_layer_call_fn_1065772???}
v?s
 ?
inputs?????????
K?H
"?
states/0?????????2
"?
states/1?????????2
p
? "c?`
?
0?????????2
A?>
?
1/0?????????2
?
1/1?????????2?
/__inference_lstm_cell_134_layer_call_fn_1065789???}
v?s
 ?
inputs?????????
K?H
"?
states/0?????????2
"?
states/1?????????2
p 
? "c?`
?
0?????????2
A?>
?
1/0?????????2
?
1/1?????????2?
K__inference_sequential_129_layer_call_and_return_conditional_losses_1064562sC?@
9?6
,?)
lstm_134_input?????????
p

 
? "%?"
?
0?????????
? ?
K__inference_sequential_129_layer_call_and_return_conditional_losses_1064578sC?@
9?6
,?)
lstm_134_input?????????
p 

 
? "%?"
?
0?????????
? ?
K__inference_sequential_129_layer_call_and_return_conditional_losses_1064825k;?8
1?.
$?!
inputs?????????
p

 
? "%?"
?
0?????????
? ?
K__inference_sequential_129_layer_call_and_return_conditional_losses_1064984k;?8
1?.
$?!
inputs?????????
p 

 
? "%?"
?
0?????????
? ?
0__inference_sequential_129_layer_call_fn_1064610fC?@
9?6
,?)
lstm_134_input?????????
p

 
? "???????????
0__inference_sequential_129_layer_call_fn_1064641fC?@
9?6
,?)
lstm_134_input?????????
p 

 
? "???????????
0__inference_sequential_129_layer_call_fn_1064999^;?8
1?.
$?!
inputs?????????
p

 
? "???????????
0__inference_sequential_129_layer_call_fn_1065014^;?8
1?.
$?!
inputs?????????
p 

 
? "???????????
%__inference_signature_wrapper_1064666?M?J
? 
C?@
>
lstm_134_input,?)
lstm_134_input?????????"5?2
0
	dense_128#? 
	dense_128?????????