
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
"serve*2.4.12v2.4.1-0-g85c8b2a817f8êÙ
x
dense_6/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:n*
shared_namedense_6/kernel
q
"dense_6/kernel/Read/ReadVariableOpReadVariableOpdense_6/kernel*
_output_shapes

:n*
dtype0
p
dense_6/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_6/bias
i
 dense_6/bias/Read/ReadVariableOpReadVariableOpdense_6/bias*
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

lstm_6/lstm_cell_6/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	¸**
shared_namelstm_6/lstm_cell_6/kernel

-lstm_6/lstm_cell_6/kernel/Read/ReadVariableOpReadVariableOplstm_6/lstm_cell_6/kernel*
_output_shapes
:	¸*
dtype0
£
#lstm_6/lstm_cell_6/recurrent_kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	n¸*4
shared_name%#lstm_6/lstm_cell_6/recurrent_kernel

7lstm_6/lstm_cell_6/recurrent_kernel/Read/ReadVariableOpReadVariableOp#lstm_6/lstm_cell_6/recurrent_kernel*
_output_shapes
:	n¸*
dtype0

lstm_6/lstm_cell_6/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:¸*(
shared_namelstm_6/lstm_cell_6/bias

+lstm_6/lstm_cell_6/bias/Read/ReadVariableOpReadVariableOplstm_6/lstm_cell_6/bias*
_output_shapes	
:¸*
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

Adam/dense_6/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:n*&
shared_nameAdam/dense_6/kernel/m

)Adam/dense_6/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_6/kernel/m*
_output_shapes

:n*
dtype0
~
Adam/dense_6/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*$
shared_nameAdam/dense_6/bias/m
w
'Adam/dense_6/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_6/bias/m*
_output_shapes
:*
dtype0

 Adam/lstm_6/lstm_cell_6/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	¸*1
shared_name" Adam/lstm_6/lstm_cell_6/kernel/m

4Adam/lstm_6/lstm_cell_6/kernel/m/Read/ReadVariableOpReadVariableOp Adam/lstm_6/lstm_cell_6/kernel/m*
_output_shapes
:	¸*
dtype0
±
*Adam/lstm_6/lstm_cell_6/recurrent_kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	n¸*;
shared_name,*Adam/lstm_6/lstm_cell_6/recurrent_kernel/m
ª
>Adam/lstm_6/lstm_cell_6/recurrent_kernel/m/Read/ReadVariableOpReadVariableOp*Adam/lstm_6/lstm_cell_6/recurrent_kernel/m*
_output_shapes
:	n¸*
dtype0

Adam/lstm_6/lstm_cell_6/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:¸*/
shared_name Adam/lstm_6/lstm_cell_6/bias/m

2Adam/lstm_6/lstm_cell_6/bias/m/Read/ReadVariableOpReadVariableOpAdam/lstm_6/lstm_cell_6/bias/m*
_output_shapes	
:¸*
dtype0

Adam/dense_6/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:n*&
shared_nameAdam/dense_6/kernel/v

)Adam/dense_6/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_6/kernel/v*
_output_shapes

:n*
dtype0
~
Adam/dense_6/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*$
shared_nameAdam/dense_6/bias/v
w
'Adam/dense_6/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_6/bias/v*
_output_shapes
:*
dtype0

 Adam/lstm_6/lstm_cell_6/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	¸*1
shared_name" Adam/lstm_6/lstm_cell_6/kernel/v

4Adam/lstm_6/lstm_cell_6/kernel/v/Read/ReadVariableOpReadVariableOp Adam/lstm_6/lstm_cell_6/kernel/v*
_output_shapes
:	¸*
dtype0
±
*Adam/lstm_6/lstm_cell_6/recurrent_kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	n¸*;
shared_name,*Adam/lstm_6/lstm_cell_6/recurrent_kernel/v
ª
>Adam/lstm_6/lstm_cell_6/recurrent_kernel/v/Read/ReadVariableOpReadVariableOp*Adam/lstm_6/lstm_cell_6/recurrent_kernel/v*
_output_shapes
:	n¸*
dtype0

Adam/lstm_6/lstm_cell_6/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:¸*/
shared_name Adam/lstm_6/lstm_cell_6/bias/v

2Adam/lstm_6/lstm_cell_6/bias/v/Read/ReadVariableOpReadVariableOpAdam/lstm_6/lstm_cell_6/bias/v*
_output_shapes	
:¸*
dtype0

NoOpNoOp
  
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*Û
valueÑBÎ BÇ
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
regularization_losses
	variables
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
layer_metrics
	variables
regularization_losses
non_trainable_variables
metrics

 layers
trainable_variables
!layer_regularization_losses
 
~

kernel
recurrent_kernel
bias
"regularization_losses
#	variables
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
&layer_metrics

'states
	variables
regularization_losses
(non_trainable_variables
)metrics

*layers
trainable_variables
+layer_regularization_losses
ZX
VARIABLE_VALUEdense_6/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE
VT
VARIABLE_VALUEdense_6/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE
 

0
1

0
1
­
,layer_metrics
regularization_losses
	variables
trainable_variables
-non_trainable_variables

.layers
/metrics
0layer_regularization_losses
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
US
VARIABLE_VALUElstm_6/lstm_cell_6/kernel&variables/0/.ATTRIBUTES/VARIABLE_VALUE
_]
VARIABLE_VALUE#lstm_6/lstm_cell_6/recurrent_kernel&variables/1/.ATTRIBUTES/VARIABLE_VALUE
SQ
VARIABLE_VALUElstm_6/lstm_cell_6/bias&variables/2/.ATTRIBUTES/VARIABLE_VALUE
 
 

10

0
1
 
 

0
1
2

0
1
2
­
2layer_metrics
"regularization_losses
#	variables
$trainable_variables
3non_trainable_variables

4layers
5metrics
6layer_regularization_losses
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
}{
VARIABLE_VALUEAdam/dense_6/kernel/mRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUEAdam/dense_6/bias/mPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
xv
VARIABLE_VALUE Adam/lstm_6/lstm_cell_6/kernel/mBvariables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE*Adam/lstm_6/lstm_cell_6/recurrent_kernel/mBvariables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
vt
VARIABLE_VALUEAdam/lstm_6/lstm_cell_6/bias/mBvariables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}{
VARIABLE_VALUEAdam/dense_6/kernel/vRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUEAdam/dense_6/bias/vPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
xv
VARIABLE_VALUE Adam/lstm_6/lstm_cell_6/kernel/vBvariables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE*Adam/lstm_6/lstm_cell_6/recurrent_kernel/vBvariables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
vt
VARIABLE_VALUEAdam/lstm_6/lstm_cell_6/bias/vBvariables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE

serving_default_lstm_6_inputPlaceholder*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype0* 
shape:ÿÿÿÿÿÿÿÿÿ
·
StatefulPartitionedCallStatefulPartitionedCallserving_default_lstm_6_inputlstm_6/lstm_cell_6/kernel#lstm_6/lstm_cell_6/recurrent_kernellstm_6/lstm_cell_6/biasdense_6/kerneldense_6/bias*
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
GPU 2J 8 *,
f'R%
#__inference_signature_wrapper_62427
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
æ	
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename"dense_6/kernel/Read/ReadVariableOp dense_6/bias/Read/ReadVariableOpAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOp&Adam/learning_rate/Read/ReadVariableOp-lstm_6/lstm_cell_6/kernel/Read/ReadVariableOp7lstm_6/lstm_cell_6/recurrent_kernel/Read/ReadVariableOp+lstm_6/lstm_cell_6/bias/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOp)Adam/dense_6/kernel/m/Read/ReadVariableOp'Adam/dense_6/bias/m/Read/ReadVariableOp4Adam/lstm_6/lstm_cell_6/kernel/m/Read/ReadVariableOp>Adam/lstm_6/lstm_cell_6/recurrent_kernel/m/Read/ReadVariableOp2Adam/lstm_6/lstm_cell_6/bias/m/Read/ReadVariableOp)Adam/dense_6/kernel/v/Read/ReadVariableOp'Adam/dense_6/bias/v/Read/ReadVariableOp4Adam/lstm_6/lstm_cell_6/kernel/v/Read/ReadVariableOp>Adam/lstm_6/lstm_cell_6/recurrent_kernel/v/Read/ReadVariableOp2Adam/lstm_6/lstm_cell_6/bias/v/Read/ReadVariableOpConst*#
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
GPU 2J 8 *'
f"R 
__inference__traced_save_63639
©
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamedense_6/kerneldense_6/bias	Adam/iterAdam/beta_1Adam/beta_2
Adam/decayAdam/learning_ratelstm_6/lstm_cell_6/kernel#lstm_6/lstm_cell_6/recurrent_kernellstm_6/lstm_cell_6/biastotalcountAdam/dense_6/kernel/mAdam/dense_6/bias/m Adam/lstm_6/lstm_cell_6/kernel/m*Adam/lstm_6/lstm_cell_6/recurrent_kernel/mAdam/lstm_6/lstm_cell_6/bias/mAdam/dense_6/kernel/vAdam/dense_6/bias/v Adam/lstm_6/lstm_cell_6/kernel/v*Adam/lstm_6/lstm_cell_6/recurrent_kernel/vAdam/lstm_6/lstm_cell_6/bias/v*"
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
GPU 2J 8 **
f%R#
!__inference__traced_restore_63715¨÷
Ý
´
,__inference_sequential_6_layer_call_fn_62371
lstm_6_input
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
identity¢StatefulPartitionedCall¤
StatefulPartitionedCallStatefulPartitionedCalllstm_6_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3*
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
GPU 2J 8 *P
fKRI
G__inference_sequential_6_layer_call_and_return_conditional_losses_623582
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:ÿÿÿÿÿÿÿÿÿ:::::22
StatefulPartitionedCallStatefulPartitionedCall:Y U
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
&
_user_specified_namelstm_6_input
¦
¾
while_cond_62180
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_13
/while_while_cond_62180___redundant_placeholder03
/while_while_cond_62180___redundant_placeholder13
/while_while_cond_62180___redundant_placeholder23
/while_while_cond_62180___redundant_placeholder3
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
@: : : : :ÿÿÿÿÿÿÿÿÿn:ÿÿÿÿÿÿÿÿÿn: ::::: 
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
:ÿÿÿÿÿÿÿÿÿn:-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿn:

_output_shapes
: :

_output_shapes
:
Õ_
¢
!__inference__traced_restore_63715
file_prefix#
assignvariableop_dense_6_kernel#
assignvariableop_1_dense_6_bias 
assignvariableop_2_adam_iter"
assignvariableop_3_adam_beta_1"
assignvariableop_4_adam_beta_2!
assignvariableop_5_adam_decay)
%assignvariableop_6_adam_learning_rate0
,assignvariableop_7_lstm_6_lstm_cell_6_kernel:
6assignvariableop_8_lstm_6_lstm_cell_6_recurrent_kernel.
*assignvariableop_9_lstm_6_lstm_cell_6_bias
assignvariableop_10_total
assignvariableop_11_count-
)assignvariableop_12_adam_dense_6_kernel_m+
'assignvariableop_13_adam_dense_6_bias_m8
4assignvariableop_14_adam_lstm_6_lstm_cell_6_kernel_mB
>assignvariableop_15_adam_lstm_6_lstm_cell_6_recurrent_kernel_m6
2assignvariableop_16_adam_lstm_6_lstm_cell_6_bias_m-
)assignvariableop_17_adam_dense_6_kernel_v+
'assignvariableop_18_adam_dense_6_bias_v8
4assignvariableop_19_adam_lstm_6_lstm_cell_6_kernel_vB
>assignvariableop_20_adam_lstm_6_lstm_cell_6_recurrent_kernel_v6
2assignvariableop_21_adam_lstm_6_lstm_cell_6_bias_v
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

Identity
AssignVariableOpAssignVariableOpassignvariableop_dense_6_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1¤
AssignVariableOp_1AssignVariableOpassignvariableop_1_dense_6_biasIdentity_1:output:0"/device:CPU:0*
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

Identity_7±
AssignVariableOp_7AssignVariableOp,assignvariableop_7_lstm_6_lstm_cell_6_kernelIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_7k

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:2

Identity_8»
AssignVariableOp_8AssignVariableOp6assignvariableop_8_lstm_6_lstm_cell_6_recurrent_kernelIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_8k

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:2

Identity_9¯
AssignVariableOp_9AssignVariableOp*assignvariableop_9_lstm_6_lstm_cell_6_biasIdentity_9:output:0"/device:CPU:0*
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
Identity_12±
AssignVariableOp_12AssignVariableOp)assignvariableop_12_adam_dense_6_kernel_mIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_12n
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:2
Identity_13¯
AssignVariableOp_13AssignVariableOp'assignvariableop_13_adam_dense_6_bias_mIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_13n
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:2
Identity_14¼
AssignVariableOp_14AssignVariableOp4assignvariableop_14_adam_lstm_6_lstm_cell_6_kernel_mIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_14n
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:2
Identity_15Æ
AssignVariableOp_15AssignVariableOp>assignvariableop_15_adam_lstm_6_lstm_cell_6_recurrent_kernel_mIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_15n
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:2
Identity_16º
AssignVariableOp_16AssignVariableOp2assignvariableop_16_adam_lstm_6_lstm_cell_6_bias_mIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_16n
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:2
Identity_17±
AssignVariableOp_17AssignVariableOp)assignvariableop_17_adam_dense_6_kernel_vIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_17n
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:2
Identity_18¯
AssignVariableOp_18AssignVariableOp'assignvariableop_18_adam_dense_6_bias_vIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_18n
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:2
Identity_19¼
AssignVariableOp_19AssignVariableOp4assignvariableop_19_adam_lstm_6_lstm_cell_6_kernel_vIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_19n
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:2
Identity_20Æ
AssignVariableOp_20AssignVariableOp>assignvariableop_20_adam_lstm_6_lstm_cell_6_recurrent_kernel_vIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_20n
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:2
Identity_21º
AssignVariableOp_21AssignVariableOp2assignvariableop_21_adam_lstm_6_lstm_cell_6_bias_vIdentity_21:output:0"/device:CPU:0*
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
ÿ

&__inference_lstm_6_layer_call_fn_63420

inputs
unknown
	unknown_0
	unknown_1
identity¢StatefulPartitionedCallþ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿn*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *J
fERC
A__inference_lstm_6_layer_call_and_return_conditional_losses_621132
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿn2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:ÿÿÿÿÿÿÿÿÿ:::22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ÚL
Ô	
lstm_6_while_body_62654*
&lstm_6_while_lstm_6_while_loop_counter0
,lstm_6_while_lstm_6_while_maximum_iterations
lstm_6_while_placeholder
lstm_6_while_placeholder_1
lstm_6_while_placeholder_2
lstm_6_while_placeholder_3)
%lstm_6_while_lstm_6_strided_slice_1_0e
alstm_6_while_tensorarrayv2read_tensorlistgetitem_lstm_6_tensorarrayunstack_tensorlistfromtensor_0=
9lstm_6_while_lstm_cell_6_matmul_readvariableop_resource_0?
;lstm_6_while_lstm_cell_6_matmul_1_readvariableop_resource_0>
:lstm_6_while_lstm_cell_6_biasadd_readvariableop_resource_0
lstm_6_while_identity
lstm_6_while_identity_1
lstm_6_while_identity_2
lstm_6_while_identity_3
lstm_6_while_identity_4
lstm_6_while_identity_5'
#lstm_6_while_lstm_6_strided_slice_1c
_lstm_6_while_tensorarrayv2read_tensorlistgetitem_lstm_6_tensorarrayunstack_tensorlistfromtensor;
7lstm_6_while_lstm_cell_6_matmul_readvariableop_resource=
9lstm_6_while_lstm_cell_6_matmul_1_readvariableop_resource<
8lstm_6_while_lstm_cell_6_biasadd_readvariableop_resource¢/lstm_6/while/lstm_cell_6/BiasAdd/ReadVariableOp¢.lstm_6/while/lstm_cell_6/MatMul/ReadVariableOp¢0lstm_6/while/lstm_cell_6/MatMul_1/ReadVariableOpÑ
>lstm_6/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   2@
>lstm_6/while/TensorArrayV2Read/TensorListGetItem/element_shapeý
0lstm_6/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemalstm_6_while_tensorarrayv2read_tensorlistgetitem_lstm_6_tensorarrayunstack_tensorlistfromtensor_0lstm_6_while_placeholderGlstm_6/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
element_dtype022
0lstm_6/while/TensorArrayV2Read/TensorListGetItemÛ
.lstm_6/while/lstm_cell_6/MatMul/ReadVariableOpReadVariableOp9lstm_6_while_lstm_cell_6_matmul_readvariableop_resource_0*
_output_shapes
:	¸*
dtype020
.lstm_6/while/lstm_cell_6/MatMul/ReadVariableOpð
lstm_6/while/lstm_cell_6/MatMulMatMul7lstm_6/while/TensorArrayV2Read/TensorListGetItem:item:06lstm_6/while/lstm_cell_6/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¸2!
lstm_6/while/lstm_cell_6/MatMulá
0lstm_6/while/lstm_cell_6/MatMul_1/ReadVariableOpReadVariableOp;lstm_6_while_lstm_cell_6_matmul_1_readvariableop_resource_0*
_output_shapes
:	n¸*
dtype022
0lstm_6/while/lstm_cell_6/MatMul_1/ReadVariableOpÙ
!lstm_6/while/lstm_cell_6/MatMul_1MatMullstm_6_while_placeholder_28lstm_6/while/lstm_cell_6/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¸2#
!lstm_6/while/lstm_cell_6/MatMul_1Ð
lstm_6/while/lstm_cell_6/addAddV2)lstm_6/while/lstm_cell_6/MatMul:product:0+lstm_6/while/lstm_cell_6/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¸2
lstm_6/while/lstm_cell_6/addÚ
/lstm_6/while/lstm_cell_6/BiasAdd/ReadVariableOpReadVariableOp:lstm_6_while_lstm_cell_6_biasadd_readvariableop_resource_0*
_output_shapes	
:¸*
dtype021
/lstm_6/while/lstm_cell_6/BiasAdd/ReadVariableOpÝ
 lstm_6/while/lstm_cell_6/BiasAddBiasAdd lstm_6/while/lstm_cell_6/add:z:07lstm_6/while/lstm_cell_6/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¸2"
 lstm_6/while/lstm_cell_6/BiasAdd
lstm_6/while/lstm_cell_6/ConstConst*
_output_shapes
: *
dtype0*
value	B :2 
lstm_6/while/lstm_cell_6/Const
(lstm_6/while/lstm_cell_6/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2*
(lstm_6/while/lstm_cell_6/split/split_dim£
lstm_6/while/lstm_cell_6/splitSplit1lstm_6/while/lstm_cell_6/split/split_dim:output:0)lstm_6/while/lstm_cell_6/BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿn:ÿÿÿÿÿÿÿÿÿn:ÿÿÿÿÿÿÿÿÿn:ÿÿÿÿÿÿÿÿÿn*
	num_split2 
lstm_6/while/lstm_cell_6/splitª
 lstm_6/while/lstm_cell_6/SigmoidSigmoid'lstm_6/while/lstm_cell_6/split:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿn2"
 lstm_6/while/lstm_cell_6/Sigmoid®
"lstm_6/while/lstm_cell_6/Sigmoid_1Sigmoid'lstm_6/while/lstm_cell_6/split:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿn2$
"lstm_6/while/lstm_cell_6/Sigmoid_1¹
lstm_6/while/lstm_cell_6/mulMul&lstm_6/while/lstm_cell_6/Sigmoid_1:y:0lstm_6_while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿn2
lstm_6/while/lstm_cell_6/mul¡
lstm_6/while/lstm_cell_6/ReluRelu'lstm_6/while/lstm_cell_6/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿn2
lstm_6/while/lstm_cell_6/ReluÌ
lstm_6/while/lstm_cell_6/mul_1Mul$lstm_6/while/lstm_cell_6/Sigmoid:y:0+lstm_6/while/lstm_cell_6/Relu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿn2 
lstm_6/while/lstm_cell_6/mul_1Á
lstm_6/while/lstm_cell_6/add_1AddV2 lstm_6/while/lstm_cell_6/mul:z:0"lstm_6/while/lstm_cell_6/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿn2 
lstm_6/while/lstm_cell_6/add_1®
"lstm_6/while/lstm_cell_6/Sigmoid_2Sigmoid'lstm_6/while/lstm_cell_6/split:output:3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿn2$
"lstm_6/while/lstm_cell_6/Sigmoid_2 
lstm_6/while/lstm_cell_6/Relu_1Relu"lstm_6/while/lstm_cell_6/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿn2!
lstm_6/while/lstm_cell_6/Relu_1Ð
lstm_6/while/lstm_cell_6/mul_2Mul&lstm_6/while/lstm_cell_6/Sigmoid_2:y:0-lstm_6/while/lstm_cell_6/Relu_1:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿn2 
lstm_6/while/lstm_cell_6/mul_2
1lstm_6/while/TensorArrayV2Write/TensorListSetItemTensorListSetItemlstm_6_while_placeholder_1lstm_6_while_placeholder"lstm_6/while/lstm_cell_6/mul_2:z:0*
_output_shapes
: *
element_dtype023
1lstm_6/while/TensorArrayV2Write/TensorListSetItemj
lstm_6/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_6/while/add/y
lstm_6/while/addAddV2lstm_6_while_placeholderlstm_6/while/add/y:output:0*
T0*
_output_shapes
: 2
lstm_6/while/addn
lstm_6/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_6/while/add_1/y
lstm_6/while/add_1AddV2&lstm_6_while_lstm_6_while_loop_counterlstm_6/while/add_1/y:output:0*
T0*
_output_shapes
: 2
lstm_6/while/add_1
lstm_6/while/IdentityIdentitylstm_6/while/add_1:z:00^lstm_6/while/lstm_cell_6/BiasAdd/ReadVariableOp/^lstm_6/while/lstm_cell_6/MatMul/ReadVariableOp1^lstm_6/while/lstm_cell_6/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
lstm_6/while/Identity£
lstm_6/while/Identity_1Identity,lstm_6_while_lstm_6_while_maximum_iterations0^lstm_6/while/lstm_cell_6/BiasAdd/ReadVariableOp/^lstm_6/while/lstm_cell_6/MatMul/ReadVariableOp1^lstm_6/while/lstm_cell_6/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
lstm_6/while/Identity_1
lstm_6/while/Identity_2Identitylstm_6/while/add:z:00^lstm_6/while/lstm_cell_6/BiasAdd/ReadVariableOp/^lstm_6/while/lstm_cell_6/MatMul/ReadVariableOp1^lstm_6/while/lstm_cell_6/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
lstm_6/while/Identity_2¸
lstm_6/while/Identity_3IdentityAlstm_6/while/TensorArrayV2Write/TensorListSetItem:output_handle:00^lstm_6/while/lstm_cell_6/BiasAdd/ReadVariableOp/^lstm_6/while/lstm_cell_6/MatMul/ReadVariableOp1^lstm_6/while/lstm_cell_6/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
lstm_6/while/Identity_3ª
lstm_6/while/Identity_4Identity"lstm_6/while/lstm_cell_6/mul_2:z:00^lstm_6/while/lstm_cell_6/BiasAdd/ReadVariableOp/^lstm_6/while/lstm_cell_6/MatMul/ReadVariableOp1^lstm_6/while/lstm_cell_6/MatMul_1/ReadVariableOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿn2
lstm_6/while/Identity_4ª
lstm_6/while/Identity_5Identity"lstm_6/while/lstm_cell_6/add_1:z:00^lstm_6/while/lstm_cell_6/BiasAdd/ReadVariableOp/^lstm_6/while/lstm_cell_6/MatMul/ReadVariableOp1^lstm_6/while/lstm_cell_6/MatMul_1/ReadVariableOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿn2
lstm_6/while/Identity_5"7
lstm_6_while_identitylstm_6/while/Identity:output:0";
lstm_6_while_identity_1 lstm_6/while/Identity_1:output:0";
lstm_6_while_identity_2 lstm_6/while/Identity_2:output:0";
lstm_6_while_identity_3 lstm_6/while/Identity_3:output:0";
lstm_6_while_identity_4 lstm_6/while/Identity_4:output:0";
lstm_6_while_identity_5 lstm_6/while/Identity_5:output:0"L
#lstm_6_while_lstm_6_strided_slice_1%lstm_6_while_lstm_6_strided_slice_1_0"v
8lstm_6_while_lstm_cell_6_biasadd_readvariableop_resource:lstm_6_while_lstm_cell_6_biasadd_readvariableop_resource_0"x
9lstm_6_while_lstm_cell_6_matmul_1_readvariableop_resource;lstm_6_while_lstm_cell_6_matmul_1_readvariableop_resource_0"t
7lstm_6_while_lstm_cell_6_matmul_readvariableop_resource9lstm_6_while_lstm_cell_6_matmul_readvariableop_resource_0"Ä
_lstm_6_while_tensorarrayv2read_tensorlistgetitem_lstm_6_tensorarrayunstack_tensorlistfromtensoralstm_6_while_tensorarrayv2read_tensorlistgetitem_lstm_6_tensorarrayunstack_tensorlistfromtensor_0*Q
_input_shapes@
>: : : : :ÿÿÿÿÿÿÿÿÿn:ÿÿÿÿÿÿÿÿÿn: : :::2b
/lstm_6/while/lstm_cell_6/BiasAdd/ReadVariableOp/lstm_6/while/lstm_cell_6/BiasAdd/ReadVariableOp2`
.lstm_6/while/lstm_cell_6/MatMul/ReadVariableOp.lstm_6/while/lstm_cell_6/MatMul/ReadVariableOp2d
0lstm_6/while/lstm_cell_6/MatMul_1/ReadVariableOp0lstm_6/while/lstm_cell_6/MatMul_1/ReadVariableOp: 
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
:ÿÿÿÿÿÿÿÿÿn:-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿn:

_output_shapes
: :

_output_shapes
: 
à

G__inference_sequential_6_layer_call_and_return_conditional_losses_62339
lstm_6_input
lstm_6_62326
lstm_6_62328
lstm_6_62330
dense_6_62333
dense_6_62335
identity¢dense_6/StatefulPartitionedCall¢lstm_6/StatefulPartitionedCall
lstm_6/StatefulPartitionedCallStatefulPartitionedCalllstm_6_inputlstm_6_62326lstm_6_62328lstm_6_62330*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿn*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *J
fERC
A__inference_lstm_6_layer_call_and_return_conditional_losses_622662 
lstm_6/StatefulPartitionedCall­
dense_6/StatefulPartitionedCallStatefulPartitionedCall'lstm_6/StatefulPartitionedCall:output:0dense_6_62333dense_6_62335*
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
GPU 2J 8 *K
fFRD
B__inference_dense_6_layer_call_and_return_conditional_losses_623062!
dense_6/StatefulPartitionedCall¿
IdentityIdentity(dense_6/StatefulPartitionedCall:output:0 ^dense_6/StatefulPartitionedCall^lstm_6/StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:ÿÿÿÿÿÿÿÿÿ:::::2B
dense_6/StatefulPartitionedCalldense_6/StatefulPartitionedCall2@
lstm_6/StatefulPartitionedCalllstm_6/StatefulPartitionedCall:Y U
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
&
_user_specified_namelstm_6_input
ì$

while_body_61879
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0
while_lstm_cell_6_61903_0
while_lstm_cell_6_61905_0
while_lstm_cell_6_61907_0
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor
while_lstm_cell_6_61903
while_lstm_cell_6_61905
while_lstm_cell_6_61907¢)while/lstm_cell_6/StatefulPartitionedCallÃ
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
)while/TensorArrayV2Read/TensorListGetItem×
)while/lstm_cell_6/StatefulPartitionedCallStatefulPartitionedCall0while/TensorArrayV2Read/TensorListGetItem:item:0while_placeholder_2while_placeholder_3while_lstm_cell_6_61903_0while_lstm_cell_6_61905_0while_lstm_cell_6_61907_0*
Tin

2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿn:ÿÿÿÿÿÿÿÿÿn:ÿÿÿÿÿÿÿÿÿn*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_lstm_cell_6_layer_call_and_return_conditional_losses_614532+
)while/lstm_cell_6/StatefulPartitionedCallö
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder2while/lstm_cell_6/StatefulPartitionedCall:output:0*
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
while/add_1
while/IdentityIdentitywhile/add_1:z:0*^while/lstm_cell_6/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity
while/Identity_1Identitywhile_while_maximum_iterations*^while/lstm_cell_6/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity_1
while/Identity_2Identitywhile/add:z:0*^while/lstm_cell_6/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity_2¹
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0*^while/lstm_cell_6/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity_3Â
while/Identity_4Identity2while/lstm_cell_6/StatefulPartitionedCall:output:1*^while/lstm_cell_6/StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿn2
while/Identity_4Â
while/Identity_5Identity2while/lstm_cell_6/StatefulPartitionedCall:output:2*^while/lstm_cell_6/StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿn2
while/Identity_5")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"4
while_lstm_cell_6_61903while_lstm_cell_6_61903_0"4
while_lstm_cell_6_61905while_lstm_cell_6_61905_0"4
while_lstm_cell_6_61907while_lstm_cell_6_61907_0"0
while_strided_slice_1while_strided_slice_1_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*Q
_input_shapes@
>: : : : :ÿÿÿÿÿÿÿÿÿn:ÿÿÿÿÿÿÿÿÿn: : :::2V
)while/lstm_cell_6/StatefulPartitionedCall)while/lstm_cell_6/StatefulPartitionedCall: 
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
:ÿÿÿÿÿÿÿÿÿn:-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿn:

_output_shapes
: :

_output_shapes
: 

©
 __inference__wrapped_model_61347
lstm_6_inputB
>sequential_6_lstm_6_lstm_cell_6_matmul_readvariableop_resourceD
@sequential_6_lstm_6_lstm_cell_6_matmul_1_readvariableop_resourceC
?sequential_6_lstm_6_lstm_cell_6_biasadd_readvariableop_resource7
3sequential_6_dense_6_matmul_readvariableop_resource8
4sequential_6_dense_6_biasadd_readvariableop_resource
identity¢+sequential_6/dense_6/BiasAdd/ReadVariableOp¢*sequential_6/dense_6/MatMul/ReadVariableOp¢6sequential_6/lstm_6/lstm_cell_6/BiasAdd/ReadVariableOp¢5sequential_6/lstm_6/lstm_cell_6/MatMul/ReadVariableOp¢7sequential_6/lstm_6/lstm_cell_6/MatMul_1/ReadVariableOp¢sequential_6/lstm_6/whiler
sequential_6/lstm_6/ShapeShapelstm_6_input*
T0*
_output_shapes
:2
sequential_6/lstm_6/Shape
'sequential_6/lstm_6/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2)
'sequential_6/lstm_6/strided_slice/stack 
)sequential_6/lstm_6/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2+
)sequential_6/lstm_6/strided_slice/stack_1 
)sequential_6/lstm_6/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2+
)sequential_6/lstm_6/strided_slice/stack_2Ú
!sequential_6/lstm_6/strided_sliceStridedSlice"sequential_6/lstm_6/Shape:output:00sequential_6/lstm_6/strided_slice/stack:output:02sequential_6/lstm_6/strided_slice/stack_1:output:02sequential_6/lstm_6/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2#
!sequential_6/lstm_6/strided_slice
sequential_6/lstm_6/zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :n2!
sequential_6/lstm_6/zeros/mul/y¼
sequential_6/lstm_6/zeros/mulMul*sequential_6/lstm_6/strided_slice:output:0(sequential_6/lstm_6/zeros/mul/y:output:0*
T0*
_output_shapes
: 2
sequential_6/lstm_6/zeros/mul
 sequential_6/lstm_6/zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :è2"
 sequential_6/lstm_6/zeros/Less/y·
sequential_6/lstm_6/zeros/LessLess!sequential_6/lstm_6/zeros/mul:z:0)sequential_6/lstm_6/zeros/Less/y:output:0*
T0*
_output_shapes
: 2 
sequential_6/lstm_6/zeros/Less
"sequential_6/lstm_6/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :n2$
"sequential_6/lstm_6/zeros/packed/1Ó
 sequential_6/lstm_6/zeros/packedPack*sequential_6/lstm_6/strided_slice:output:0+sequential_6/lstm_6/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2"
 sequential_6/lstm_6/zeros/packed
sequential_6/lstm_6/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2!
sequential_6/lstm_6/zeros/ConstÅ
sequential_6/lstm_6/zerosFill)sequential_6/lstm_6/zeros/packed:output:0(sequential_6/lstm_6/zeros/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿn2
sequential_6/lstm_6/zeros
!sequential_6/lstm_6/zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B :n2#
!sequential_6/lstm_6/zeros_1/mul/yÂ
sequential_6/lstm_6/zeros_1/mulMul*sequential_6/lstm_6/strided_slice:output:0*sequential_6/lstm_6/zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2!
sequential_6/lstm_6/zeros_1/mul
"sequential_6/lstm_6/zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :è2$
"sequential_6/lstm_6/zeros_1/Less/y¿
 sequential_6/lstm_6/zeros_1/LessLess#sequential_6/lstm_6/zeros_1/mul:z:0+sequential_6/lstm_6/zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2"
 sequential_6/lstm_6/zeros_1/Less
$sequential_6/lstm_6/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :n2&
$sequential_6/lstm_6/zeros_1/packed/1Ù
"sequential_6/lstm_6/zeros_1/packedPack*sequential_6/lstm_6/strided_slice:output:0-sequential_6/lstm_6/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2$
"sequential_6/lstm_6/zeros_1/packed
!sequential_6/lstm_6/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2#
!sequential_6/lstm_6/zeros_1/ConstÍ
sequential_6/lstm_6/zeros_1Fill+sequential_6/lstm_6/zeros_1/packed:output:0*sequential_6/lstm_6/zeros_1/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿn2
sequential_6/lstm_6/zeros_1
"sequential_6/lstm_6/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2$
"sequential_6/lstm_6/transpose/perm¼
sequential_6/lstm_6/transpose	Transposelstm_6_input+sequential_6/lstm_6/transpose/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
sequential_6/lstm_6/transpose
sequential_6/lstm_6/Shape_1Shape!sequential_6/lstm_6/transpose:y:0*
T0*
_output_shapes
:2
sequential_6/lstm_6/Shape_1 
)sequential_6/lstm_6/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2+
)sequential_6/lstm_6/strided_slice_1/stack¤
+sequential_6/lstm_6/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2-
+sequential_6/lstm_6/strided_slice_1/stack_1¤
+sequential_6/lstm_6/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2-
+sequential_6/lstm_6/strided_slice_1/stack_2æ
#sequential_6/lstm_6/strided_slice_1StridedSlice$sequential_6/lstm_6/Shape_1:output:02sequential_6/lstm_6/strided_slice_1/stack:output:04sequential_6/lstm_6/strided_slice_1/stack_1:output:04sequential_6/lstm_6/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2%
#sequential_6/lstm_6/strided_slice_1­
/sequential_6/lstm_6/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ21
/sequential_6/lstm_6/TensorArrayV2/element_shape
!sequential_6/lstm_6/TensorArrayV2TensorListReserve8sequential_6/lstm_6/TensorArrayV2/element_shape:output:0,sequential_6/lstm_6/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02#
!sequential_6/lstm_6/TensorArrayV2ç
Isequential_6/lstm_6/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   2K
Isequential_6/lstm_6/TensorArrayUnstack/TensorListFromTensor/element_shapeÈ
;sequential_6/lstm_6/TensorArrayUnstack/TensorListFromTensorTensorListFromTensor!sequential_6/lstm_6/transpose:y:0Rsequential_6/lstm_6/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02=
;sequential_6/lstm_6/TensorArrayUnstack/TensorListFromTensor 
)sequential_6/lstm_6/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2+
)sequential_6/lstm_6/strided_slice_2/stack¤
+sequential_6/lstm_6/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2-
+sequential_6/lstm_6/strided_slice_2/stack_1¤
+sequential_6/lstm_6/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2-
+sequential_6/lstm_6/strided_slice_2/stack_2ô
#sequential_6/lstm_6/strided_slice_2StridedSlice!sequential_6/lstm_6/transpose:y:02sequential_6/lstm_6/strided_slice_2/stack:output:04sequential_6/lstm_6/strided_slice_2/stack_1:output:04sequential_6/lstm_6/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
shrink_axis_mask2%
#sequential_6/lstm_6/strided_slice_2î
5sequential_6/lstm_6/lstm_cell_6/MatMul/ReadVariableOpReadVariableOp>sequential_6_lstm_6_lstm_cell_6_matmul_readvariableop_resource*
_output_shapes
:	¸*
dtype027
5sequential_6/lstm_6/lstm_cell_6/MatMul/ReadVariableOpú
&sequential_6/lstm_6/lstm_cell_6/MatMulMatMul,sequential_6/lstm_6/strided_slice_2:output:0=sequential_6/lstm_6/lstm_cell_6/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¸2(
&sequential_6/lstm_6/lstm_cell_6/MatMulô
7sequential_6/lstm_6/lstm_cell_6/MatMul_1/ReadVariableOpReadVariableOp@sequential_6_lstm_6_lstm_cell_6_matmul_1_readvariableop_resource*
_output_shapes
:	n¸*
dtype029
7sequential_6/lstm_6/lstm_cell_6/MatMul_1/ReadVariableOpö
(sequential_6/lstm_6/lstm_cell_6/MatMul_1MatMul"sequential_6/lstm_6/zeros:output:0?sequential_6/lstm_6/lstm_cell_6/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¸2*
(sequential_6/lstm_6/lstm_cell_6/MatMul_1ì
#sequential_6/lstm_6/lstm_cell_6/addAddV20sequential_6/lstm_6/lstm_cell_6/MatMul:product:02sequential_6/lstm_6/lstm_cell_6/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¸2%
#sequential_6/lstm_6/lstm_cell_6/addí
6sequential_6/lstm_6/lstm_cell_6/BiasAdd/ReadVariableOpReadVariableOp?sequential_6_lstm_6_lstm_cell_6_biasadd_readvariableop_resource*
_output_shapes	
:¸*
dtype028
6sequential_6/lstm_6/lstm_cell_6/BiasAdd/ReadVariableOpù
'sequential_6/lstm_6/lstm_cell_6/BiasAddBiasAdd'sequential_6/lstm_6/lstm_cell_6/add:z:0>sequential_6/lstm_6/lstm_cell_6/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¸2)
'sequential_6/lstm_6/lstm_cell_6/BiasAdd
%sequential_6/lstm_6/lstm_cell_6/ConstConst*
_output_shapes
: *
dtype0*
value	B :2'
%sequential_6/lstm_6/lstm_cell_6/Const¤
/sequential_6/lstm_6/lstm_cell_6/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :21
/sequential_6/lstm_6/lstm_cell_6/split/split_dim¿
%sequential_6/lstm_6/lstm_cell_6/splitSplit8sequential_6/lstm_6/lstm_cell_6/split/split_dim:output:00sequential_6/lstm_6/lstm_cell_6/BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿn:ÿÿÿÿÿÿÿÿÿn:ÿÿÿÿÿÿÿÿÿn:ÿÿÿÿÿÿÿÿÿn*
	num_split2'
%sequential_6/lstm_6/lstm_cell_6/split¿
'sequential_6/lstm_6/lstm_cell_6/SigmoidSigmoid.sequential_6/lstm_6/lstm_cell_6/split:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿn2)
'sequential_6/lstm_6/lstm_cell_6/SigmoidÃ
)sequential_6/lstm_6/lstm_cell_6/Sigmoid_1Sigmoid.sequential_6/lstm_6/lstm_cell_6/split:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿn2+
)sequential_6/lstm_6/lstm_cell_6/Sigmoid_1Ø
#sequential_6/lstm_6/lstm_cell_6/mulMul-sequential_6/lstm_6/lstm_cell_6/Sigmoid_1:y:0$sequential_6/lstm_6/zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿn2%
#sequential_6/lstm_6/lstm_cell_6/mul¶
$sequential_6/lstm_6/lstm_cell_6/ReluRelu.sequential_6/lstm_6/lstm_cell_6/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿn2&
$sequential_6/lstm_6/lstm_cell_6/Reluè
%sequential_6/lstm_6/lstm_cell_6/mul_1Mul+sequential_6/lstm_6/lstm_cell_6/Sigmoid:y:02sequential_6/lstm_6/lstm_cell_6/Relu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿn2'
%sequential_6/lstm_6/lstm_cell_6/mul_1Ý
%sequential_6/lstm_6/lstm_cell_6/add_1AddV2'sequential_6/lstm_6/lstm_cell_6/mul:z:0)sequential_6/lstm_6/lstm_cell_6/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿn2'
%sequential_6/lstm_6/lstm_cell_6/add_1Ã
)sequential_6/lstm_6/lstm_cell_6/Sigmoid_2Sigmoid.sequential_6/lstm_6/lstm_cell_6/split:output:3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿn2+
)sequential_6/lstm_6/lstm_cell_6/Sigmoid_2µ
&sequential_6/lstm_6/lstm_cell_6/Relu_1Relu)sequential_6/lstm_6/lstm_cell_6/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿn2(
&sequential_6/lstm_6/lstm_cell_6/Relu_1ì
%sequential_6/lstm_6/lstm_cell_6/mul_2Mul-sequential_6/lstm_6/lstm_cell_6/Sigmoid_2:y:04sequential_6/lstm_6/lstm_cell_6/Relu_1:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿn2'
%sequential_6/lstm_6/lstm_cell_6/mul_2·
1sequential_6/lstm_6/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿn   23
1sequential_6/lstm_6/TensorArrayV2_1/element_shape
#sequential_6/lstm_6/TensorArrayV2_1TensorListReserve:sequential_6/lstm_6/TensorArrayV2_1/element_shape:output:0,sequential_6/lstm_6/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02%
#sequential_6/lstm_6/TensorArrayV2_1v
sequential_6/lstm_6/timeConst*
_output_shapes
: *
dtype0*
value	B : 2
sequential_6/lstm_6/time§
,sequential_6/lstm_6/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2.
,sequential_6/lstm_6/while/maximum_iterations
&sequential_6/lstm_6/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2(
&sequential_6/lstm_6/while/loop_counter
sequential_6/lstm_6/whileWhile/sequential_6/lstm_6/while/loop_counter:output:05sequential_6/lstm_6/while/maximum_iterations:output:0!sequential_6/lstm_6/time:output:0,sequential_6/lstm_6/TensorArrayV2_1:handle:0"sequential_6/lstm_6/zeros:output:0$sequential_6/lstm_6/zeros_1:output:0,sequential_6/lstm_6/strided_slice_1:output:0Ksequential_6/lstm_6/TensorArrayUnstack/TensorListFromTensor:output_handle:0>sequential_6_lstm_6_lstm_cell_6_matmul_readvariableop_resource@sequential_6_lstm_6_lstm_cell_6_matmul_1_readvariableop_resource?sequential_6_lstm_6_lstm_cell_6_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :ÿÿÿÿÿÿÿÿÿn:ÿÿÿÿÿÿÿÿÿn: : : : : *%
_read_only_resource_inputs
	
*0
body(R&
$sequential_6_lstm_6_while_body_61256*0
cond(R&
$sequential_6_lstm_6_while_cond_61255*K
output_shapes:
8: : : : :ÿÿÿÿÿÿÿÿÿn:ÿÿÿÿÿÿÿÿÿn: : : : : *
parallel_iterations 2
sequential_6/lstm_6/whileÝ
Dsequential_6/lstm_6/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿn   2F
Dsequential_6/lstm_6/TensorArrayV2Stack/TensorListStack/element_shape¸
6sequential_6/lstm_6/TensorArrayV2Stack/TensorListStackTensorListStack"sequential_6/lstm_6/while:output:3Msequential_6/lstm_6/TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿn*
element_dtype028
6sequential_6/lstm_6/TensorArrayV2Stack/TensorListStack©
)sequential_6/lstm_6/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿ2+
)sequential_6/lstm_6/strided_slice_3/stack¤
+sequential_6/lstm_6/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2-
+sequential_6/lstm_6/strided_slice_3/stack_1¤
+sequential_6/lstm_6/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2-
+sequential_6/lstm_6/strided_slice_3/stack_2
#sequential_6/lstm_6/strided_slice_3StridedSlice?sequential_6/lstm_6/TensorArrayV2Stack/TensorListStack:tensor:02sequential_6/lstm_6/strided_slice_3/stack:output:04sequential_6/lstm_6/strided_slice_3/stack_1:output:04sequential_6/lstm_6/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿn*
shrink_axis_mask2%
#sequential_6/lstm_6/strided_slice_3¡
$sequential_6/lstm_6/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2&
$sequential_6/lstm_6/transpose_1/permõ
sequential_6/lstm_6/transpose_1	Transpose?sequential_6/lstm_6/TensorArrayV2Stack/TensorListStack:tensor:0-sequential_6/lstm_6/transpose_1/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿn2!
sequential_6/lstm_6/transpose_1
sequential_6/lstm_6/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2
sequential_6/lstm_6/runtimeÌ
*sequential_6/dense_6/MatMul/ReadVariableOpReadVariableOp3sequential_6_dense_6_matmul_readvariableop_resource*
_output_shapes

:n*
dtype02,
*sequential_6/dense_6/MatMul/ReadVariableOpØ
sequential_6/dense_6/MatMulMatMul,sequential_6/lstm_6/strided_slice_3:output:02sequential_6/dense_6/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
sequential_6/dense_6/MatMulË
+sequential_6/dense_6/BiasAdd/ReadVariableOpReadVariableOp4sequential_6_dense_6_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02-
+sequential_6/dense_6/BiasAdd/ReadVariableOpÕ
sequential_6/dense_6/BiasAddBiasAdd%sequential_6/dense_6/MatMul:product:03sequential_6/dense_6/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
sequential_6/dense_6/BiasAdd
IdentityIdentity%sequential_6/dense_6/BiasAdd:output:0,^sequential_6/dense_6/BiasAdd/ReadVariableOp+^sequential_6/dense_6/MatMul/ReadVariableOp7^sequential_6/lstm_6/lstm_cell_6/BiasAdd/ReadVariableOp6^sequential_6/lstm_6/lstm_cell_6/MatMul/ReadVariableOp8^sequential_6/lstm_6/lstm_cell_6/MatMul_1/ReadVariableOp^sequential_6/lstm_6/while*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:ÿÿÿÿÿÿÿÿÿ:::::2Z
+sequential_6/dense_6/BiasAdd/ReadVariableOp+sequential_6/dense_6/BiasAdd/ReadVariableOp2X
*sequential_6/dense_6/MatMul/ReadVariableOp*sequential_6/dense_6/MatMul/ReadVariableOp2p
6sequential_6/lstm_6/lstm_cell_6/BiasAdd/ReadVariableOp6sequential_6/lstm_6/lstm_cell_6/BiasAdd/ReadVariableOp2n
5sequential_6/lstm_6/lstm_cell_6/MatMul/ReadVariableOp5sequential_6/lstm_6/lstm_cell_6/MatMul/ReadVariableOp2r
7sequential_6/lstm_6/lstm_cell_6/MatMul_1/ReadVariableOp7sequential_6/lstm_6/lstm_cell_6/MatMul_1/ReadVariableOp26
sequential_6/lstm_6/whilesequential_6/lstm_6/while:Y U
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
&
_user_specified_namelstm_6_input
¦
¾
while_cond_61878
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_13
/while_while_cond_61878___redundant_placeholder03
/while_while_cond_61878___redundant_placeholder13
/while_while_cond_61878___redundant_placeholder23
/while_while_cond_61878___redundant_placeholder3
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
@: : : : :ÿÿÿÿÿÿÿÿÿn:ÿÿÿÿÿÿÿÿÿn: ::::: 
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
:ÿÿÿÿÿÿÿÿÿn:-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿn:

_output_shapes
: :

_output_shapes
:
ì$

while_body_61747
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0
while_lstm_cell_6_61771_0
while_lstm_cell_6_61773_0
while_lstm_cell_6_61775_0
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor
while_lstm_cell_6_61771
while_lstm_cell_6_61773
while_lstm_cell_6_61775¢)while/lstm_cell_6/StatefulPartitionedCallÃ
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
)while/TensorArrayV2Read/TensorListGetItem×
)while/lstm_cell_6/StatefulPartitionedCallStatefulPartitionedCall0while/TensorArrayV2Read/TensorListGetItem:item:0while_placeholder_2while_placeholder_3while_lstm_cell_6_61771_0while_lstm_cell_6_61773_0while_lstm_cell_6_61775_0*
Tin

2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿn:ÿÿÿÿÿÿÿÿÿn:ÿÿÿÿÿÿÿÿÿn*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_lstm_cell_6_layer_call_and_return_conditional_losses_614202+
)while/lstm_cell_6/StatefulPartitionedCallö
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder2while/lstm_cell_6/StatefulPartitionedCall:output:0*
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
while/add_1
while/IdentityIdentitywhile/add_1:z:0*^while/lstm_cell_6/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity
while/Identity_1Identitywhile_while_maximum_iterations*^while/lstm_cell_6/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity_1
while/Identity_2Identitywhile/add:z:0*^while/lstm_cell_6/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity_2¹
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0*^while/lstm_cell_6/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity_3Â
while/Identity_4Identity2while/lstm_cell_6/StatefulPartitionedCall:output:1*^while/lstm_cell_6/StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿn2
while/Identity_4Â
while/Identity_5Identity2while/lstm_cell_6/StatefulPartitionedCall:output:2*^while/lstm_cell_6/StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿn2
while/Identity_5")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"4
while_lstm_cell_6_61771while_lstm_cell_6_61771_0"4
while_lstm_cell_6_61773while_lstm_cell_6_61773_0"4
while_lstm_cell_6_61775while_lstm_cell_6_61775_0"0
while_strided_slice_1while_strided_slice_1_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*Q
_input_shapes@
>: : : : :ÿÿÿÿÿÿÿÿÿn:ÿÿÿÿÿÿÿÿÿn: : :::2V
)while/lstm_cell_6/StatefulPartitionedCall)while/lstm_cell_6/StatefulPartitionedCall: 
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
:ÿÿÿÿÿÿÿÿÿn:-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿn:

_output_shapes
: :

_output_shapes
: 
Ë
®
,__inference_sequential_6_layer_call_fn_62760

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
identity¢StatefulPartitionedCall
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
GPU 2J 8 *P
fKRI
G__inference_sequential_6_layer_call_and_return_conditional_losses_623582
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:ÿÿÿÿÿÿÿÿÿ:::::22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
­
«
#__inference_signature_wrapper_62427
lstm_6_input
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
identity¢StatefulPartitionedCallý
StatefulPartitionedCallStatefulPartitionedCalllstm_6_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3*
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
GPU 2J 8 *)
f$R"
 __inference__wrapped_model_613472
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:ÿÿÿÿÿÿÿÿÿ:::::22
StatefulPartitionedCallStatefulPartitionedCall:Y U
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
&
_user_specified_namelstm_6_input
±
Û
F__inference_lstm_cell_6_layer_call_and_return_conditional_losses_63516

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
:	¸*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¸2
MatMul
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource*
_output_shapes
:	n¸*
dtype02
MatMul_1/ReadVariableOp|
MatMul_1MatMulstates_0MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¸2

MatMul_1l
addAddV2MatMul:product:0MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¸2
add
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:¸*
dtype02
BiasAdd/ReadVariableOpy
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¸2	
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
L:ÿÿÿÿÿÿÿÿÿn:ÿÿÿÿÿÿÿÿÿn:ÿÿÿÿÿÿÿÿÿn:ÿÿÿÿÿÿÿÿÿn*
	num_split2
split_
SigmoidSigmoidsplit:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿn2	
Sigmoidc
	Sigmoid_1Sigmoidsplit:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿn2
	Sigmoid_1\
mulMulSigmoid_1:y:0states_1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿn2
mulV
ReluRelusplit:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿn2
Reluh
mul_1MulSigmoid:y:0Relu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿn2
mul_1]
add_1AddV2mul:z:0	mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿn2
add_1c
	Sigmoid_2Sigmoidsplit:output:3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿn2
	Sigmoid_2U
Relu_1Relu	add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿn2
Relu_1l
mul_2MulSigmoid_2:y:0Relu_1:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿn2
mul_2¨
IdentityIdentity	mul_2:z:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿn2

Identity¬

Identity_1Identity	mul_2:z:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿn2

Identity_1¬

Identity_2Identity	add_1:z:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿn2

Identity_2"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*X
_input_shapesG
E:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿn:ÿÿÿÿÿÿÿÿÿn:::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs:QM
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿn
"
_user_specified_name
states/0:QM
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿn
"
_user_specified_name
states/1
±
Û
F__inference_lstm_cell_6_layer_call_and_return_conditional_losses_63483

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
:	¸*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¸2
MatMul
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource*
_output_shapes
:	n¸*
dtype02
MatMul_1/ReadVariableOp|
MatMul_1MatMulstates_0MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¸2

MatMul_1l
addAddV2MatMul:product:0MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¸2
add
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:¸*
dtype02
BiasAdd/ReadVariableOpy
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¸2	
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
L:ÿÿÿÿÿÿÿÿÿn:ÿÿÿÿÿÿÿÿÿn:ÿÿÿÿÿÿÿÿÿn:ÿÿÿÿÿÿÿÿÿn*
	num_split2
split_
SigmoidSigmoidsplit:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿn2	
Sigmoidc
	Sigmoid_1Sigmoidsplit:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿn2
	Sigmoid_1\
mulMulSigmoid_1:y:0states_1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿn2
mulV
ReluRelusplit:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿn2
Reluh
mul_1MulSigmoid:y:0Relu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿn2
mul_1]
add_1AddV2mul:z:0	mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿn2
add_1c
	Sigmoid_2Sigmoidsplit:output:3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿn2
	Sigmoid_2U
Relu_1Relu	add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿn2
Relu_1l
mul_2MulSigmoid_2:y:0Relu_1:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿn2
mul_2¨
IdentityIdentity	mul_2:z:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿn2

Identity¬

Identity_1Identity	mul_2:z:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿn2

Identity_1¬

Identity_2Identity	add_1:z:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿn2

Identity_2"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*X
_input_shapesG
E:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿn:ÿÿÿÿÿÿÿÿÿn:::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs:QM
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿn
"
_user_specified_name
states/0:QM
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿn
"
_user_specified_name
states/1
Î

G__inference_sequential_6_layer_call_and_return_conditional_losses_62358

inputs
lstm_6_62345
lstm_6_62347
lstm_6_62349
dense_6_62352
dense_6_62354
identity¢dense_6/StatefulPartitionedCall¢lstm_6/StatefulPartitionedCall
lstm_6/StatefulPartitionedCallStatefulPartitionedCallinputslstm_6_62345lstm_6_62347lstm_6_62349*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿn*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *J
fERC
A__inference_lstm_6_layer_call_and_return_conditional_losses_621132 
lstm_6/StatefulPartitionedCall­
dense_6/StatefulPartitionedCallStatefulPartitionedCall'lstm_6/StatefulPartitionedCall:output:0dense_6_62352dense_6_62354*
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
GPU 2J 8 *K
fFRD
B__inference_dense_6_layer_call_and_return_conditional_losses_623062!
dense_6/StatefulPartitionedCall¿
IdentityIdentity(dense_6/StatefulPartitionedCall:output:0 ^dense_6/StatefulPartitionedCall^lstm_6/StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:ÿÿÿÿÿÿÿÿÿ:::::2B
dense_6/StatefulPartitionedCalldense_6/StatefulPartitionedCall2@
lstm_6/StatefulPartitionedCalllstm_6/StatefulPartitionedCall:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ùo
»
G__inference_sequential_6_layer_call_and_return_conditional_losses_62745

inputs5
1lstm_6_lstm_cell_6_matmul_readvariableop_resource7
3lstm_6_lstm_cell_6_matmul_1_readvariableop_resource6
2lstm_6_lstm_cell_6_biasadd_readvariableop_resource*
&dense_6_matmul_readvariableop_resource+
'dense_6_biasadd_readvariableop_resource
identity¢dense_6/BiasAdd/ReadVariableOp¢dense_6/MatMul/ReadVariableOp¢)lstm_6/lstm_cell_6/BiasAdd/ReadVariableOp¢(lstm_6/lstm_cell_6/MatMul/ReadVariableOp¢*lstm_6/lstm_cell_6/MatMul_1/ReadVariableOp¢lstm_6/whileR
lstm_6/ShapeShapeinputs*
T0*
_output_shapes
:2
lstm_6/Shape
lstm_6/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
lstm_6/strided_slice/stack
lstm_6/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
lstm_6/strided_slice/stack_1
lstm_6/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
lstm_6/strided_slice/stack_2
lstm_6/strided_sliceStridedSlicelstm_6/Shape:output:0#lstm_6/strided_slice/stack:output:0%lstm_6/strided_slice/stack_1:output:0%lstm_6/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
lstm_6/strided_slicej
lstm_6/zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :n2
lstm_6/zeros/mul/y
lstm_6/zeros/mulMullstm_6/strided_slice:output:0lstm_6/zeros/mul/y:output:0*
T0*
_output_shapes
: 2
lstm_6/zeros/mulm
lstm_6/zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :è2
lstm_6/zeros/Less/y
lstm_6/zeros/LessLesslstm_6/zeros/mul:z:0lstm_6/zeros/Less/y:output:0*
T0*
_output_shapes
: 2
lstm_6/zeros/Lessp
lstm_6/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :n2
lstm_6/zeros/packed/1
lstm_6/zeros/packedPacklstm_6/strided_slice:output:0lstm_6/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
lstm_6/zeros/packedm
lstm_6/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
lstm_6/zeros/Const
lstm_6/zerosFilllstm_6/zeros/packed:output:0lstm_6/zeros/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿn2
lstm_6/zerosn
lstm_6/zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B :n2
lstm_6/zeros_1/mul/y
lstm_6/zeros_1/mulMullstm_6/strided_slice:output:0lstm_6/zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
lstm_6/zeros_1/mulq
lstm_6/zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :è2
lstm_6/zeros_1/Less/y
lstm_6/zeros_1/LessLesslstm_6/zeros_1/mul:z:0lstm_6/zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
lstm_6/zeros_1/Lesst
lstm_6/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :n2
lstm_6/zeros_1/packed/1¥
lstm_6/zeros_1/packedPacklstm_6/strided_slice:output:0 lstm_6/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
lstm_6/zeros_1/packedq
lstm_6/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
lstm_6/zeros_1/Const
lstm_6/zeros_1Filllstm_6/zeros_1/packed:output:0lstm_6/zeros_1/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿn2
lstm_6/zeros_1
lstm_6/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
lstm_6/transpose/perm
lstm_6/transpose	Transposeinputslstm_6/transpose/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_6/transposed
lstm_6/Shape_1Shapelstm_6/transpose:y:0*
T0*
_output_shapes
:2
lstm_6/Shape_1
lstm_6/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
lstm_6/strided_slice_1/stack
lstm_6/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2 
lstm_6/strided_slice_1/stack_1
lstm_6/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2 
lstm_6/strided_slice_1/stack_2
lstm_6/strided_slice_1StridedSlicelstm_6/Shape_1:output:0%lstm_6/strided_slice_1/stack:output:0'lstm_6/strided_slice_1/stack_1:output:0'lstm_6/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
lstm_6/strided_slice_1
"lstm_6/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2$
"lstm_6/TensorArrayV2/element_shapeÎ
lstm_6/TensorArrayV2TensorListReserve+lstm_6/TensorArrayV2/element_shape:output:0lstm_6/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
lstm_6/TensorArrayV2Í
<lstm_6/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   2>
<lstm_6/TensorArrayUnstack/TensorListFromTensor/element_shape
.lstm_6/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorlstm_6/transpose:y:0Elstm_6/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type020
.lstm_6/TensorArrayUnstack/TensorListFromTensor
lstm_6/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
lstm_6/strided_slice_2/stack
lstm_6/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2 
lstm_6/strided_slice_2/stack_1
lstm_6/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2 
lstm_6/strided_slice_2/stack_2¦
lstm_6/strided_slice_2StridedSlicelstm_6/transpose:y:0%lstm_6/strided_slice_2/stack:output:0'lstm_6/strided_slice_2/stack_1:output:0'lstm_6/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
shrink_axis_mask2
lstm_6/strided_slice_2Ç
(lstm_6/lstm_cell_6/MatMul/ReadVariableOpReadVariableOp1lstm_6_lstm_cell_6_matmul_readvariableop_resource*
_output_shapes
:	¸*
dtype02*
(lstm_6/lstm_cell_6/MatMul/ReadVariableOpÆ
lstm_6/lstm_cell_6/MatMulMatMullstm_6/strided_slice_2:output:00lstm_6/lstm_cell_6/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¸2
lstm_6/lstm_cell_6/MatMulÍ
*lstm_6/lstm_cell_6/MatMul_1/ReadVariableOpReadVariableOp3lstm_6_lstm_cell_6_matmul_1_readvariableop_resource*
_output_shapes
:	n¸*
dtype02,
*lstm_6/lstm_cell_6/MatMul_1/ReadVariableOpÂ
lstm_6/lstm_cell_6/MatMul_1MatMullstm_6/zeros:output:02lstm_6/lstm_cell_6/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¸2
lstm_6/lstm_cell_6/MatMul_1¸
lstm_6/lstm_cell_6/addAddV2#lstm_6/lstm_cell_6/MatMul:product:0%lstm_6/lstm_cell_6/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¸2
lstm_6/lstm_cell_6/addÆ
)lstm_6/lstm_cell_6/BiasAdd/ReadVariableOpReadVariableOp2lstm_6_lstm_cell_6_biasadd_readvariableop_resource*
_output_shapes	
:¸*
dtype02+
)lstm_6/lstm_cell_6/BiasAdd/ReadVariableOpÅ
lstm_6/lstm_cell_6/BiasAddBiasAddlstm_6/lstm_cell_6/add:z:01lstm_6/lstm_cell_6/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¸2
lstm_6/lstm_cell_6/BiasAddv
lstm_6/lstm_cell_6/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_6/lstm_cell_6/Const
"lstm_6/lstm_cell_6/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2$
"lstm_6/lstm_cell_6/split/split_dim
lstm_6/lstm_cell_6/splitSplit+lstm_6/lstm_cell_6/split/split_dim:output:0#lstm_6/lstm_cell_6/BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿn:ÿÿÿÿÿÿÿÿÿn:ÿÿÿÿÿÿÿÿÿn:ÿÿÿÿÿÿÿÿÿn*
	num_split2
lstm_6/lstm_cell_6/split
lstm_6/lstm_cell_6/SigmoidSigmoid!lstm_6/lstm_cell_6/split:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿn2
lstm_6/lstm_cell_6/Sigmoid
lstm_6/lstm_cell_6/Sigmoid_1Sigmoid!lstm_6/lstm_cell_6/split:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿn2
lstm_6/lstm_cell_6/Sigmoid_1¤
lstm_6/lstm_cell_6/mulMul lstm_6/lstm_cell_6/Sigmoid_1:y:0lstm_6/zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿn2
lstm_6/lstm_cell_6/mul
lstm_6/lstm_cell_6/ReluRelu!lstm_6/lstm_cell_6/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿn2
lstm_6/lstm_cell_6/Relu´
lstm_6/lstm_cell_6/mul_1Mullstm_6/lstm_cell_6/Sigmoid:y:0%lstm_6/lstm_cell_6/Relu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿn2
lstm_6/lstm_cell_6/mul_1©
lstm_6/lstm_cell_6/add_1AddV2lstm_6/lstm_cell_6/mul:z:0lstm_6/lstm_cell_6/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿn2
lstm_6/lstm_cell_6/add_1
lstm_6/lstm_cell_6/Sigmoid_2Sigmoid!lstm_6/lstm_cell_6/split:output:3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿn2
lstm_6/lstm_cell_6/Sigmoid_2
lstm_6/lstm_cell_6/Relu_1Relulstm_6/lstm_cell_6/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿn2
lstm_6/lstm_cell_6/Relu_1¸
lstm_6/lstm_cell_6/mul_2Mul lstm_6/lstm_cell_6/Sigmoid_2:y:0'lstm_6/lstm_cell_6/Relu_1:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿn2
lstm_6/lstm_cell_6/mul_2
$lstm_6/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿn   2&
$lstm_6/TensorArrayV2_1/element_shapeÔ
lstm_6/TensorArrayV2_1TensorListReserve-lstm_6/TensorArrayV2_1/element_shape:output:0lstm_6/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
lstm_6/TensorArrayV2_1\
lstm_6/timeConst*
_output_shapes
: *
dtype0*
value	B : 2
lstm_6/time
lstm_6/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2!
lstm_6/while/maximum_iterationsx
lstm_6/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
lstm_6/while/loop_counterÒ
lstm_6/whileWhile"lstm_6/while/loop_counter:output:0(lstm_6/while/maximum_iterations:output:0lstm_6/time:output:0lstm_6/TensorArrayV2_1:handle:0lstm_6/zeros:output:0lstm_6/zeros_1:output:0lstm_6/strided_slice_1:output:0>lstm_6/TensorArrayUnstack/TensorListFromTensor:output_handle:01lstm_6_lstm_cell_6_matmul_readvariableop_resource3lstm_6_lstm_cell_6_matmul_1_readvariableop_resource2lstm_6_lstm_cell_6_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :ÿÿÿÿÿÿÿÿÿn:ÿÿÿÿÿÿÿÿÿn: : : : : *%
_read_only_resource_inputs
	
*#
bodyR
lstm_6_while_body_62654*#
condR
lstm_6_while_cond_62653*K
output_shapes:
8: : : : :ÿÿÿÿÿÿÿÿÿn:ÿÿÿÿÿÿÿÿÿn: : : : : *
parallel_iterations 2
lstm_6/whileÃ
7lstm_6/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿn   29
7lstm_6/TensorArrayV2Stack/TensorListStack/element_shape
)lstm_6/TensorArrayV2Stack/TensorListStackTensorListStacklstm_6/while:output:3@lstm_6/TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿn*
element_dtype02+
)lstm_6/TensorArrayV2Stack/TensorListStack
lstm_6/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿ2
lstm_6/strided_slice_3/stack
lstm_6/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2 
lstm_6/strided_slice_3/stack_1
lstm_6/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2 
lstm_6/strided_slice_3/stack_2Ä
lstm_6/strided_slice_3StridedSlice2lstm_6/TensorArrayV2Stack/TensorListStack:tensor:0%lstm_6/strided_slice_3/stack:output:0'lstm_6/strided_slice_3/stack_1:output:0'lstm_6/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿn*
shrink_axis_mask2
lstm_6/strided_slice_3
lstm_6/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
lstm_6/transpose_1/permÁ
lstm_6/transpose_1	Transpose2lstm_6/TensorArrayV2Stack/TensorListStack:tensor:0 lstm_6/transpose_1/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿn2
lstm_6/transpose_1t
lstm_6/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2
lstm_6/runtime¥
dense_6/MatMul/ReadVariableOpReadVariableOp&dense_6_matmul_readvariableop_resource*
_output_shapes

:n*
dtype02
dense_6/MatMul/ReadVariableOp¤
dense_6/MatMulMatMullstm_6/strided_slice_3:output:0%dense_6/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_6/MatMul¤
dense_6/BiasAdd/ReadVariableOpReadVariableOp'dense_6_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02 
dense_6/BiasAdd/ReadVariableOp¡
dense_6/BiasAddBiasAdddense_6/MatMul:product:0&dense_6/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_6/BiasAddÀ
IdentityIdentitydense_6/BiasAdd:output:0^dense_6/BiasAdd/ReadVariableOp^dense_6/MatMul/ReadVariableOp*^lstm_6/lstm_cell_6/BiasAdd/ReadVariableOp)^lstm_6/lstm_cell_6/MatMul/ReadVariableOp+^lstm_6/lstm_cell_6/MatMul_1/ReadVariableOp^lstm_6/while*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:ÿÿÿÿÿÿÿÿÿ:::::2@
dense_6/BiasAdd/ReadVariableOpdense_6/BiasAdd/ReadVariableOp2>
dense_6/MatMul/ReadVariableOpdense_6/MatMul/ReadVariableOp2V
)lstm_6/lstm_cell_6/BiasAdd/ReadVariableOp)lstm_6/lstm_cell_6/BiasAdd/ReadVariableOp2T
(lstm_6/lstm_cell_6/MatMul/ReadVariableOp(lstm_6/lstm_cell_6/MatMul/ReadVariableOp2X
*lstm_6/lstm_cell_6/MatMul_1/ReadVariableOp*lstm_6/lstm_cell_6/MatMul_1/ReadVariableOp2
lstm_6/whilelstm_6/while:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ôZ
ì
A__inference_lstm_6_layer_call_and_return_conditional_losses_63081
inputs_0.
*lstm_cell_6_matmul_readvariableop_resource0
,lstm_cell_6_matmul_1_readvariableop_resource/
+lstm_cell_6_biasadd_readvariableop_resource
identity¢"lstm_cell_6/BiasAdd/ReadVariableOp¢!lstm_cell_6/MatMul/ReadVariableOp¢#lstm_cell_6/MatMul_1/ReadVariableOp¢whileF
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
value	B :n2
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
value	B :n2
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
:ÿÿÿÿÿÿÿÿÿn2
zeros`
zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B :n2
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
value	B :n2
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
:ÿÿÿÿÿÿÿÿÿn2	
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
strided_slice_2²
!lstm_cell_6/MatMul/ReadVariableOpReadVariableOp*lstm_cell_6_matmul_readvariableop_resource*
_output_shapes
:	¸*
dtype02#
!lstm_cell_6/MatMul/ReadVariableOpª
lstm_cell_6/MatMulMatMulstrided_slice_2:output:0)lstm_cell_6/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¸2
lstm_cell_6/MatMul¸
#lstm_cell_6/MatMul_1/ReadVariableOpReadVariableOp,lstm_cell_6_matmul_1_readvariableop_resource*
_output_shapes
:	n¸*
dtype02%
#lstm_cell_6/MatMul_1/ReadVariableOp¦
lstm_cell_6/MatMul_1MatMulzeros:output:0+lstm_cell_6/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¸2
lstm_cell_6/MatMul_1
lstm_cell_6/addAddV2lstm_cell_6/MatMul:product:0lstm_cell_6/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¸2
lstm_cell_6/add±
"lstm_cell_6/BiasAdd/ReadVariableOpReadVariableOp+lstm_cell_6_biasadd_readvariableop_resource*
_output_shapes	
:¸*
dtype02$
"lstm_cell_6/BiasAdd/ReadVariableOp©
lstm_cell_6/BiasAddBiasAddlstm_cell_6/add:z:0*lstm_cell_6/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¸2
lstm_cell_6/BiasAddh
lstm_cell_6/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_6/Const|
lstm_cell_6/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_6/split/split_dimï
lstm_cell_6/splitSplit$lstm_cell_6/split/split_dim:output:0lstm_cell_6/BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿn:ÿÿÿÿÿÿÿÿÿn:ÿÿÿÿÿÿÿÿÿn:ÿÿÿÿÿÿÿÿÿn*
	num_split2
lstm_cell_6/split
lstm_cell_6/SigmoidSigmoidlstm_cell_6/split:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿn2
lstm_cell_6/Sigmoid
lstm_cell_6/Sigmoid_1Sigmoidlstm_cell_6/split:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿn2
lstm_cell_6/Sigmoid_1
lstm_cell_6/mulMullstm_cell_6/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿn2
lstm_cell_6/mulz
lstm_cell_6/ReluRelulstm_cell_6/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿn2
lstm_cell_6/Relu
lstm_cell_6/mul_1Mullstm_cell_6/Sigmoid:y:0lstm_cell_6/Relu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿn2
lstm_cell_6/mul_1
lstm_cell_6/add_1AddV2lstm_cell_6/mul:z:0lstm_cell_6/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿn2
lstm_cell_6/add_1
lstm_cell_6/Sigmoid_2Sigmoidlstm_cell_6/split:output:3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿn2
lstm_cell_6/Sigmoid_2y
lstm_cell_6/Relu_1Relulstm_cell_6/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿn2
lstm_cell_6/Relu_1
lstm_cell_6/mul_2Mullstm_cell_6/Sigmoid_2:y:0 lstm_cell_6/Relu_1:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿn2
lstm_cell_6/mul_2
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿn   2
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
while/loop_counteré
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0*lstm_cell_6_matmul_readvariableop_resource,lstm_cell_6_matmul_1_readvariableop_resource+lstm_cell_6_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :ÿÿÿÿÿÿÿÿÿn:ÿÿÿÿÿÿÿÿÿn: : : : : *%
_read_only_resource_inputs
	
*
bodyR
while_body_62996*
condR
while_cond_62995*K
output_shapes:
8: : : : :ÿÿÿÿÿÿÿÿÿn:ÿÿÿÿÿÿÿÿÿn: : : : : *
parallel_iterations 2
whileµ
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿn   22
0TensorArrayV2Stack/TensorListStack/element_shapeñ
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿn*
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
:ÿÿÿÿÿÿÿÿÿn*
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
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿn2
transpose_1f
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2	
runtimeã
IdentityIdentitystrided_slice_3:output:0#^lstm_cell_6/BiasAdd/ReadVariableOp"^lstm_cell_6/MatMul/ReadVariableOp$^lstm_cell_6/MatMul_1/ReadVariableOp^while*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿn2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:::2H
"lstm_cell_6/BiasAdd/ReadVariableOp"lstm_cell_6/BiasAdd/ReadVariableOp2F
!lstm_cell_6/MatMul/ReadVariableOp!lstm_cell_6/MatMul/ReadVariableOp2J
#lstm_cell_6/MatMul_1/ReadVariableOp#lstm_cell_6/MatMul_1/ReadVariableOp2
whilewhile:^ Z
4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
inputs/0
ôZ
ì
A__inference_lstm_6_layer_call_and_return_conditional_losses_62928
inputs_0.
*lstm_cell_6_matmul_readvariableop_resource0
,lstm_cell_6_matmul_1_readvariableop_resource/
+lstm_cell_6_biasadd_readvariableop_resource
identity¢"lstm_cell_6/BiasAdd/ReadVariableOp¢!lstm_cell_6/MatMul/ReadVariableOp¢#lstm_cell_6/MatMul_1/ReadVariableOp¢whileF
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
value	B :n2
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
value	B :n2
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
:ÿÿÿÿÿÿÿÿÿn2
zeros`
zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B :n2
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
value	B :n2
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
:ÿÿÿÿÿÿÿÿÿn2	
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
strided_slice_2²
!lstm_cell_6/MatMul/ReadVariableOpReadVariableOp*lstm_cell_6_matmul_readvariableop_resource*
_output_shapes
:	¸*
dtype02#
!lstm_cell_6/MatMul/ReadVariableOpª
lstm_cell_6/MatMulMatMulstrided_slice_2:output:0)lstm_cell_6/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¸2
lstm_cell_6/MatMul¸
#lstm_cell_6/MatMul_1/ReadVariableOpReadVariableOp,lstm_cell_6_matmul_1_readvariableop_resource*
_output_shapes
:	n¸*
dtype02%
#lstm_cell_6/MatMul_1/ReadVariableOp¦
lstm_cell_6/MatMul_1MatMulzeros:output:0+lstm_cell_6/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¸2
lstm_cell_6/MatMul_1
lstm_cell_6/addAddV2lstm_cell_6/MatMul:product:0lstm_cell_6/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¸2
lstm_cell_6/add±
"lstm_cell_6/BiasAdd/ReadVariableOpReadVariableOp+lstm_cell_6_biasadd_readvariableop_resource*
_output_shapes	
:¸*
dtype02$
"lstm_cell_6/BiasAdd/ReadVariableOp©
lstm_cell_6/BiasAddBiasAddlstm_cell_6/add:z:0*lstm_cell_6/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¸2
lstm_cell_6/BiasAddh
lstm_cell_6/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_6/Const|
lstm_cell_6/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_6/split/split_dimï
lstm_cell_6/splitSplit$lstm_cell_6/split/split_dim:output:0lstm_cell_6/BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿn:ÿÿÿÿÿÿÿÿÿn:ÿÿÿÿÿÿÿÿÿn:ÿÿÿÿÿÿÿÿÿn*
	num_split2
lstm_cell_6/split
lstm_cell_6/SigmoidSigmoidlstm_cell_6/split:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿn2
lstm_cell_6/Sigmoid
lstm_cell_6/Sigmoid_1Sigmoidlstm_cell_6/split:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿn2
lstm_cell_6/Sigmoid_1
lstm_cell_6/mulMullstm_cell_6/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿn2
lstm_cell_6/mulz
lstm_cell_6/ReluRelulstm_cell_6/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿn2
lstm_cell_6/Relu
lstm_cell_6/mul_1Mullstm_cell_6/Sigmoid:y:0lstm_cell_6/Relu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿn2
lstm_cell_6/mul_1
lstm_cell_6/add_1AddV2lstm_cell_6/mul:z:0lstm_cell_6/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿn2
lstm_cell_6/add_1
lstm_cell_6/Sigmoid_2Sigmoidlstm_cell_6/split:output:3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿn2
lstm_cell_6/Sigmoid_2y
lstm_cell_6/Relu_1Relulstm_cell_6/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿn2
lstm_cell_6/Relu_1
lstm_cell_6/mul_2Mullstm_cell_6/Sigmoid_2:y:0 lstm_cell_6/Relu_1:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿn2
lstm_cell_6/mul_2
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿn   2
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
while/loop_counteré
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0*lstm_cell_6_matmul_readvariableop_resource,lstm_cell_6_matmul_1_readvariableop_resource+lstm_cell_6_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :ÿÿÿÿÿÿÿÿÿn:ÿÿÿÿÿÿÿÿÿn: : : : : *%
_read_only_resource_inputs
	
*
bodyR
while_body_62843*
condR
while_cond_62842*K
output_shapes:
8: : : : :ÿÿÿÿÿÿÿÿÿn:ÿÿÿÿÿÿÿÿÿn: : : : : *
parallel_iterations 2
whileµ
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿn   22
0TensorArrayV2Stack/TensorListStack/element_shapeñ
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿn*
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
:ÿÿÿÿÿÿÿÿÿn*
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
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿn2
transpose_1f
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2	
runtimeã
IdentityIdentitystrided_slice_3:output:0#^lstm_cell_6/BiasAdd/ReadVariableOp"^lstm_cell_6/MatMul/ReadVariableOp$^lstm_cell_6/MatMul_1/ReadVariableOp^while*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿn2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:::2H
"lstm_cell_6/BiasAdd/ReadVariableOp"lstm_cell_6/BiasAdd/ReadVariableOp2F
!lstm_cell_6/MatMul/ReadVariableOp!lstm_cell_6/MatMul/ReadVariableOp2J
#lstm_cell_6/MatMul_1/ReadVariableOp#lstm_cell_6/MatMul_1/ReadVariableOp2
whilewhile:^ Z
4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
inputs/0
¾Z
ê
A__inference_lstm_6_layer_call_and_return_conditional_losses_63256

inputs.
*lstm_cell_6_matmul_readvariableop_resource0
,lstm_cell_6_matmul_1_readvariableop_resource/
+lstm_cell_6_biasadd_readvariableop_resource
identity¢"lstm_cell_6/BiasAdd/ReadVariableOp¢!lstm_cell_6/MatMul/ReadVariableOp¢#lstm_cell_6/MatMul_1/ReadVariableOp¢whileD
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
value	B :n2
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
value	B :n2
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
:ÿÿÿÿÿÿÿÿÿn2
zeros`
zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B :n2
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
value	B :n2
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
:ÿÿÿÿÿÿÿÿÿn2	
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
:ÿÿÿÿÿÿÿÿÿ2
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
strided_slice_2²
!lstm_cell_6/MatMul/ReadVariableOpReadVariableOp*lstm_cell_6_matmul_readvariableop_resource*
_output_shapes
:	¸*
dtype02#
!lstm_cell_6/MatMul/ReadVariableOpª
lstm_cell_6/MatMulMatMulstrided_slice_2:output:0)lstm_cell_6/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¸2
lstm_cell_6/MatMul¸
#lstm_cell_6/MatMul_1/ReadVariableOpReadVariableOp,lstm_cell_6_matmul_1_readvariableop_resource*
_output_shapes
:	n¸*
dtype02%
#lstm_cell_6/MatMul_1/ReadVariableOp¦
lstm_cell_6/MatMul_1MatMulzeros:output:0+lstm_cell_6/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¸2
lstm_cell_6/MatMul_1
lstm_cell_6/addAddV2lstm_cell_6/MatMul:product:0lstm_cell_6/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¸2
lstm_cell_6/add±
"lstm_cell_6/BiasAdd/ReadVariableOpReadVariableOp+lstm_cell_6_biasadd_readvariableop_resource*
_output_shapes	
:¸*
dtype02$
"lstm_cell_6/BiasAdd/ReadVariableOp©
lstm_cell_6/BiasAddBiasAddlstm_cell_6/add:z:0*lstm_cell_6/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¸2
lstm_cell_6/BiasAddh
lstm_cell_6/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_6/Const|
lstm_cell_6/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_6/split/split_dimï
lstm_cell_6/splitSplit$lstm_cell_6/split/split_dim:output:0lstm_cell_6/BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿn:ÿÿÿÿÿÿÿÿÿn:ÿÿÿÿÿÿÿÿÿn:ÿÿÿÿÿÿÿÿÿn*
	num_split2
lstm_cell_6/split
lstm_cell_6/SigmoidSigmoidlstm_cell_6/split:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿn2
lstm_cell_6/Sigmoid
lstm_cell_6/Sigmoid_1Sigmoidlstm_cell_6/split:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿn2
lstm_cell_6/Sigmoid_1
lstm_cell_6/mulMullstm_cell_6/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿn2
lstm_cell_6/mulz
lstm_cell_6/ReluRelulstm_cell_6/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿn2
lstm_cell_6/Relu
lstm_cell_6/mul_1Mullstm_cell_6/Sigmoid:y:0lstm_cell_6/Relu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿn2
lstm_cell_6/mul_1
lstm_cell_6/add_1AddV2lstm_cell_6/mul:z:0lstm_cell_6/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿn2
lstm_cell_6/add_1
lstm_cell_6/Sigmoid_2Sigmoidlstm_cell_6/split:output:3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿn2
lstm_cell_6/Sigmoid_2y
lstm_cell_6/Relu_1Relulstm_cell_6/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿn2
lstm_cell_6/Relu_1
lstm_cell_6/mul_2Mullstm_cell_6/Sigmoid_2:y:0 lstm_cell_6/Relu_1:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿn2
lstm_cell_6/mul_2
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿn   2
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
while/loop_counteré
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0*lstm_cell_6_matmul_readvariableop_resource,lstm_cell_6_matmul_1_readvariableop_resource+lstm_cell_6_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :ÿÿÿÿÿÿÿÿÿn:ÿÿÿÿÿÿÿÿÿn: : : : : *%
_read_only_resource_inputs
	
*
bodyR
while_body_63171*
condR
while_cond_63170*K
output_shapes:
8: : : : :ÿÿÿÿÿÿÿÿÿn:ÿÿÿÿÿÿÿÿÿn: : : : : *
parallel_iterations 2
whileµ
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿn   22
0TensorArrayV2Stack/TensorListStack/element_shapeè
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿn*
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
:ÿÿÿÿÿÿÿÿÿn*
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
:ÿÿÿÿÿÿÿÿÿn2
transpose_1f
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2	
runtimeã
IdentityIdentitystrided_slice_3:output:0#^lstm_cell_6/BiasAdd/ReadVariableOp"^lstm_cell_6/MatMul/ReadVariableOp$^lstm_cell_6/MatMul_1/ReadVariableOp^while*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿn2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:ÿÿÿÿÿÿÿÿÿ:::2H
"lstm_cell_6/BiasAdd/ReadVariableOp"lstm_cell_6/BiasAdd/ReadVariableOp2F
!lstm_cell_6/MatMul/ReadVariableOp!lstm_cell_6/MatMul/ReadVariableOp2J
#lstm_cell_6/MatMul_1/ReadVariableOp#lstm_cell_6/MatMul_1/ReadVariableOp2
whilewhile:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
©
Ù
F__inference_lstm_cell_6_layer_call_and_return_conditional_losses_61453

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
:	¸*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¸2
MatMul
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource*
_output_shapes
:	n¸*
dtype02
MatMul_1/ReadVariableOpz
MatMul_1MatMulstatesMatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¸2

MatMul_1l
addAddV2MatMul:product:0MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¸2
add
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:¸*
dtype02
BiasAdd/ReadVariableOpy
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¸2	
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
L:ÿÿÿÿÿÿÿÿÿn:ÿÿÿÿÿÿÿÿÿn:ÿÿÿÿÿÿÿÿÿn:ÿÿÿÿÿÿÿÿÿn*
	num_split2
split_
SigmoidSigmoidsplit:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿn2	
Sigmoidc
	Sigmoid_1Sigmoidsplit:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿn2
	Sigmoid_1\
mulMulSigmoid_1:y:0states_1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿn2
mulV
ReluRelusplit:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿn2
Reluh
mul_1MulSigmoid:y:0Relu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿn2
mul_1]
add_1AddV2mul:z:0	mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿn2
add_1c
	Sigmoid_2Sigmoidsplit:output:3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿn2
	Sigmoid_2U
Relu_1Relu	add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿn2
Relu_1l
mul_2MulSigmoid_2:y:0Relu_1:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿn2
mul_2¨
IdentityIdentity	mul_2:z:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿn2

Identity¬

Identity_1Identity	mul_2:z:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿn2

Identity_1¬

Identity_2Identity	add_1:z:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿn2

Identity_2"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*X
_input_shapesG
E:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿn:ÿÿÿÿÿÿÿÿÿn:::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs:OK
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿn
 
_user_specified_namestates:OK
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿn
 
_user_specified_namestates
¾Z
ê
A__inference_lstm_6_layer_call_and_return_conditional_losses_62113

inputs.
*lstm_cell_6_matmul_readvariableop_resource0
,lstm_cell_6_matmul_1_readvariableop_resource/
+lstm_cell_6_biasadd_readvariableop_resource
identity¢"lstm_cell_6/BiasAdd/ReadVariableOp¢!lstm_cell_6/MatMul/ReadVariableOp¢#lstm_cell_6/MatMul_1/ReadVariableOp¢whileD
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
value	B :n2
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
value	B :n2
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
:ÿÿÿÿÿÿÿÿÿn2
zeros`
zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B :n2
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
value	B :n2
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
:ÿÿÿÿÿÿÿÿÿn2	
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
:ÿÿÿÿÿÿÿÿÿ2
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
strided_slice_2²
!lstm_cell_6/MatMul/ReadVariableOpReadVariableOp*lstm_cell_6_matmul_readvariableop_resource*
_output_shapes
:	¸*
dtype02#
!lstm_cell_6/MatMul/ReadVariableOpª
lstm_cell_6/MatMulMatMulstrided_slice_2:output:0)lstm_cell_6/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¸2
lstm_cell_6/MatMul¸
#lstm_cell_6/MatMul_1/ReadVariableOpReadVariableOp,lstm_cell_6_matmul_1_readvariableop_resource*
_output_shapes
:	n¸*
dtype02%
#lstm_cell_6/MatMul_1/ReadVariableOp¦
lstm_cell_6/MatMul_1MatMulzeros:output:0+lstm_cell_6/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¸2
lstm_cell_6/MatMul_1
lstm_cell_6/addAddV2lstm_cell_6/MatMul:product:0lstm_cell_6/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¸2
lstm_cell_6/add±
"lstm_cell_6/BiasAdd/ReadVariableOpReadVariableOp+lstm_cell_6_biasadd_readvariableop_resource*
_output_shapes	
:¸*
dtype02$
"lstm_cell_6/BiasAdd/ReadVariableOp©
lstm_cell_6/BiasAddBiasAddlstm_cell_6/add:z:0*lstm_cell_6/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¸2
lstm_cell_6/BiasAddh
lstm_cell_6/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_6/Const|
lstm_cell_6/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_6/split/split_dimï
lstm_cell_6/splitSplit$lstm_cell_6/split/split_dim:output:0lstm_cell_6/BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿn:ÿÿÿÿÿÿÿÿÿn:ÿÿÿÿÿÿÿÿÿn:ÿÿÿÿÿÿÿÿÿn*
	num_split2
lstm_cell_6/split
lstm_cell_6/SigmoidSigmoidlstm_cell_6/split:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿn2
lstm_cell_6/Sigmoid
lstm_cell_6/Sigmoid_1Sigmoidlstm_cell_6/split:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿn2
lstm_cell_6/Sigmoid_1
lstm_cell_6/mulMullstm_cell_6/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿn2
lstm_cell_6/mulz
lstm_cell_6/ReluRelulstm_cell_6/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿn2
lstm_cell_6/Relu
lstm_cell_6/mul_1Mullstm_cell_6/Sigmoid:y:0lstm_cell_6/Relu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿn2
lstm_cell_6/mul_1
lstm_cell_6/add_1AddV2lstm_cell_6/mul:z:0lstm_cell_6/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿn2
lstm_cell_6/add_1
lstm_cell_6/Sigmoid_2Sigmoidlstm_cell_6/split:output:3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿn2
lstm_cell_6/Sigmoid_2y
lstm_cell_6/Relu_1Relulstm_cell_6/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿn2
lstm_cell_6/Relu_1
lstm_cell_6/mul_2Mullstm_cell_6/Sigmoid_2:y:0 lstm_cell_6/Relu_1:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿn2
lstm_cell_6/mul_2
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿn   2
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
while/loop_counteré
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0*lstm_cell_6_matmul_readvariableop_resource,lstm_cell_6_matmul_1_readvariableop_resource+lstm_cell_6_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :ÿÿÿÿÿÿÿÿÿn:ÿÿÿÿÿÿÿÿÿn: : : : : *%
_read_only_resource_inputs
	
*
bodyR
while_body_62028*
condR
while_cond_62027*K
output_shapes:
8: : : : :ÿÿÿÿÿÿÿÿÿn:ÿÿÿÿÿÿÿÿÿn: : : : : *
parallel_iterations 2
whileµ
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿn   22
0TensorArrayV2Stack/TensorListStack/element_shapeè
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿn*
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
:ÿÿÿÿÿÿÿÿÿn*
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
:ÿÿÿÿÿÿÿÿÿn2
transpose_1f
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2	
runtimeã
IdentityIdentitystrided_slice_3:output:0#^lstm_cell_6/BiasAdd/ReadVariableOp"^lstm_cell_6/MatMul/ReadVariableOp$^lstm_cell_6/MatMul_1/ReadVariableOp^while*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿn2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:ÿÿÿÿÿÿÿÿÿ:::2H
"lstm_cell_6/BiasAdd/ReadVariableOp"lstm_cell_6/BiasAdd/ReadVariableOp2F
!lstm_cell_6/MatMul/ReadVariableOp!lstm_cell_6/MatMul/ReadVariableOp2J
#lstm_cell_6/MatMul_1/ReadVariableOp#lstm_cell_6/MatMul_1/ReadVariableOp2
whilewhile:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
¦
¾
while_cond_63323
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_13
/while_while_cond_63323___redundant_placeholder03
/while_while_cond_63323___redundant_placeholder13
/while_while_cond_63323___redundant_placeholder23
/while_while_cond_63323___redundant_placeholder3
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
@: : : : :ÿÿÿÿÿÿÿÿÿn:ÿÿÿÿÿÿÿÿÿn: ::::: 
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
:ÿÿÿÿÿÿÿÿÿn:-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿn:

_output_shapes
: :

_output_shapes
:
¦
¾
while_cond_61746
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_13
/while_while_cond_61746___redundant_placeholder03
/while_while_cond_61746___redundant_placeholder13
/while_while_cond_61746___redundant_placeholder23
/while_while_cond_61746___redundant_placeholder3
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
@: : : : :ÿÿÿÿÿÿÿÿÿn:ÿÿÿÿÿÿÿÿÿn: ::::: 
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
:ÿÿÿÿÿÿÿÿÿn:-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿn:

_output_shapes
: :

_output_shapes
:
¾Z
ê
A__inference_lstm_6_layer_call_and_return_conditional_losses_63409

inputs.
*lstm_cell_6_matmul_readvariableop_resource0
,lstm_cell_6_matmul_1_readvariableop_resource/
+lstm_cell_6_biasadd_readvariableop_resource
identity¢"lstm_cell_6/BiasAdd/ReadVariableOp¢!lstm_cell_6/MatMul/ReadVariableOp¢#lstm_cell_6/MatMul_1/ReadVariableOp¢whileD
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
value	B :n2
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
value	B :n2
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
:ÿÿÿÿÿÿÿÿÿn2
zeros`
zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B :n2
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
value	B :n2
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
:ÿÿÿÿÿÿÿÿÿn2	
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
:ÿÿÿÿÿÿÿÿÿ2
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
strided_slice_2²
!lstm_cell_6/MatMul/ReadVariableOpReadVariableOp*lstm_cell_6_matmul_readvariableop_resource*
_output_shapes
:	¸*
dtype02#
!lstm_cell_6/MatMul/ReadVariableOpª
lstm_cell_6/MatMulMatMulstrided_slice_2:output:0)lstm_cell_6/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¸2
lstm_cell_6/MatMul¸
#lstm_cell_6/MatMul_1/ReadVariableOpReadVariableOp,lstm_cell_6_matmul_1_readvariableop_resource*
_output_shapes
:	n¸*
dtype02%
#lstm_cell_6/MatMul_1/ReadVariableOp¦
lstm_cell_6/MatMul_1MatMulzeros:output:0+lstm_cell_6/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¸2
lstm_cell_6/MatMul_1
lstm_cell_6/addAddV2lstm_cell_6/MatMul:product:0lstm_cell_6/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¸2
lstm_cell_6/add±
"lstm_cell_6/BiasAdd/ReadVariableOpReadVariableOp+lstm_cell_6_biasadd_readvariableop_resource*
_output_shapes	
:¸*
dtype02$
"lstm_cell_6/BiasAdd/ReadVariableOp©
lstm_cell_6/BiasAddBiasAddlstm_cell_6/add:z:0*lstm_cell_6/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¸2
lstm_cell_6/BiasAddh
lstm_cell_6/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_6/Const|
lstm_cell_6/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_6/split/split_dimï
lstm_cell_6/splitSplit$lstm_cell_6/split/split_dim:output:0lstm_cell_6/BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿn:ÿÿÿÿÿÿÿÿÿn:ÿÿÿÿÿÿÿÿÿn:ÿÿÿÿÿÿÿÿÿn*
	num_split2
lstm_cell_6/split
lstm_cell_6/SigmoidSigmoidlstm_cell_6/split:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿn2
lstm_cell_6/Sigmoid
lstm_cell_6/Sigmoid_1Sigmoidlstm_cell_6/split:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿn2
lstm_cell_6/Sigmoid_1
lstm_cell_6/mulMullstm_cell_6/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿn2
lstm_cell_6/mulz
lstm_cell_6/ReluRelulstm_cell_6/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿn2
lstm_cell_6/Relu
lstm_cell_6/mul_1Mullstm_cell_6/Sigmoid:y:0lstm_cell_6/Relu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿn2
lstm_cell_6/mul_1
lstm_cell_6/add_1AddV2lstm_cell_6/mul:z:0lstm_cell_6/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿn2
lstm_cell_6/add_1
lstm_cell_6/Sigmoid_2Sigmoidlstm_cell_6/split:output:3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿn2
lstm_cell_6/Sigmoid_2y
lstm_cell_6/Relu_1Relulstm_cell_6/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿn2
lstm_cell_6/Relu_1
lstm_cell_6/mul_2Mullstm_cell_6/Sigmoid_2:y:0 lstm_cell_6/Relu_1:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿn2
lstm_cell_6/mul_2
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿn   2
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
while/loop_counteré
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0*lstm_cell_6_matmul_readvariableop_resource,lstm_cell_6_matmul_1_readvariableop_resource+lstm_cell_6_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :ÿÿÿÿÿÿÿÿÿn:ÿÿÿÿÿÿÿÿÿn: : : : : *%
_read_only_resource_inputs
	
*
bodyR
while_body_63324*
condR
while_cond_63323*K
output_shapes:
8: : : : :ÿÿÿÿÿÿÿÿÿn:ÿÿÿÿÿÿÿÿÿn: : : : : *
parallel_iterations 2
whileµ
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿn   22
0TensorArrayV2Stack/TensorListStack/element_shapeè
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿn*
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
:ÿÿÿÿÿÿÿÿÿn*
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
:ÿÿÿÿÿÿÿÿÿn2
transpose_1f
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2	
runtimeã
IdentityIdentitystrided_slice_3:output:0#^lstm_cell_6/BiasAdd/ReadVariableOp"^lstm_cell_6/MatMul/ReadVariableOp$^lstm_cell_6/MatMul_1/ReadVariableOp^while*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿn2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:ÿÿÿÿÿÿÿÿÿ:::2H
"lstm_cell_6/BiasAdd/ReadVariableOp"lstm_cell_6/BiasAdd/ReadVariableOp2F
!lstm_cell_6/MatMul/ReadVariableOp!lstm_cell_6/MatMul/ReadVariableOp2J
#lstm_cell_6/MatMul_1/ReadVariableOp#lstm_cell_6/MatMul_1/ReadVariableOp2
whilewhile:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ø
|
'__inference_dense_6_layer_call_fn_63450

inputs
unknown
	unknown_0
identity¢StatefulPartitionedCallò
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
GPU 2J 8 *K
fFRD
B__inference_dense_6_layer_call_and_return_conditional_losses_623062
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿn::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿn
 
_user_specified_nameinputs
	
Û
B__inference_dense_6_layer_call_and_return_conditional_losses_62306

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:n*
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
:ÿÿÿÿÿÿÿÿÿn::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿn
 
_user_specified_nameinputs
B
ô
while_body_62028
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_06
2while_lstm_cell_6_matmul_readvariableop_resource_08
4while_lstm_cell_6_matmul_1_readvariableop_resource_07
3while_lstm_cell_6_biasadd_readvariableop_resource_0
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor4
0while_lstm_cell_6_matmul_readvariableop_resource6
2while_lstm_cell_6_matmul_1_readvariableop_resource5
1while_lstm_cell_6_biasadd_readvariableop_resource¢(while/lstm_cell_6/BiasAdd/ReadVariableOp¢'while/lstm_cell_6/MatMul/ReadVariableOp¢)while/lstm_cell_6/MatMul_1/ReadVariableOpÃ
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
)while/TensorArrayV2Read/TensorListGetItemÆ
'while/lstm_cell_6/MatMul/ReadVariableOpReadVariableOp2while_lstm_cell_6_matmul_readvariableop_resource_0*
_output_shapes
:	¸*
dtype02)
'while/lstm_cell_6/MatMul/ReadVariableOpÔ
while/lstm_cell_6/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0/while/lstm_cell_6/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¸2
while/lstm_cell_6/MatMulÌ
)while/lstm_cell_6/MatMul_1/ReadVariableOpReadVariableOp4while_lstm_cell_6_matmul_1_readvariableop_resource_0*
_output_shapes
:	n¸*
dtype02+
)while/lstm_cell_6/MatMul_1/ReadVariableOp½
while/lstm_cell_6/MatMul_1MatMulwhile_placeholder_21while/lstm_cell_6/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¸2
while/lstm_cell_6/MatMul_1´
while/lstm_cell_6/addAddV2"while/lstm_cell_6/MatMul:product:0$while/lstm_cell_6/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¸2
while/lstm_cell_6/addÅ
(while/lstm_cell_6/BiasAdd/ReadVariableOpReadVariableOp3while_lstm_cell_6_biasadd_readvariableop_resource_0*
_output_shapes	
:¸*
dtype02*
(while/lstm_cell_6/BiasAdd/ReadVariableOpÁ
while/lstm_cell_6/BiasAddBiasAddwhile/lstm_cell_6/add:z:00while/lstm_cell_6/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¸2
while/lstm_cell_6/BiasAddt
while/lstm_cell_6/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
while/lstm_cell_6/Const
!while/lstm_cell_6/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2#
!while/lstm_cell_6/split/split_dim
while/lstm_cell_6/splitSplit*while/lstm_cell_6/split/split_dim:output:0"while/lstm_cell_6/BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿn:ÿÿÿÿÿÿÿÿÿn:ÿÿÿÿÿÿÿÿÿn:ÿÿÿÿÿÿÿÿÿn*
	num_split2
while/lstm_cell_6/split
while/lstm_cell_6/SigmoidSigmoid while/lstm_cell_6/split:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿn2
while/lstm_cell_6/Sigmoid
while/lstm_cell_6/Sigmoid_1Sigmoid while/lstm_cell_6/split:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿn2
while/lstm_cell_6/Sigmoid_1
while/lstm_cell_6/mulMulwhile/lstm_cell_6/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿn2
while/lstm_cell_6/mul
while/lstm_cell_6/ReluRelu while/lstm_cell_6/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿn2
while/lstm_cell_6/Relu°
while/lstm_cell_6/mul_1Mulwhile/lstm_cell_6/Sigmoid:y:0$while/lstm_cell_6/Relu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿn2
while/lstm_cell_6/mul_1¥
while/lstm_cell_6/add_1AddV2while/lstm_cell_6/mul:z:0while/lstm_cell_6/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿn2
while/lstm_cell_6/add_1
while/lstm_cell_6/Sigmoid_2Sigmoid while/lstm_cell_6/split:output:3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿn2
while/lstm_cell_6/Sigmoid_2
while/lstm_cell_6/Relu_1Reluwhile/lstm_cell_6/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿn2
while/lstm_cell_6/Relu_1´
while/lstm_cell_6/mul_2Mulwhile/lstm_cell_6/Sigmoid_2:y:0&while/lstm_cell_6/Relu_1:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿn2
while/lstm_cell_6/mul_2ß
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_6/mul_2:z:0*
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
while/add_1ß
while/IdentityIdentitywhile/add_1:z:0)^while/lstm_cell_6/BiasAdd/ReadVariableOp(^while/lstm_cell_6/MatMul/ReadVariableOp*^while/lstm_cell_6/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identityò
while/Identity_1Identitywhile_while_maximum_iterations)^while/lstm_cell_6/BiasAdd/ReadVariableOp(^while/lstm_cell_6/MatMul/ReadVariableOp*^while/lstm_cell_6/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_1á
while/Identity_2Identitywhile/add:z:0)^while/lstm_cell_6/BiasAdd/ReadVariableOp(^while/lstm_cell_6/MatMul/ReadVariableOp*^while/lstm_cell_6/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_2
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0)^while/lstm_cell_6/BiasAdd/ReadVariableOp(^while/lstm_cell_6/MatMul/ReadVariableOp*^while/lstm_cell_6/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_3
while/Identity_4Identitywhile/lstm_cell_6/mul_2:z:0)^while/lstm_cell_6/BiasAdd/ReadVariableOp(^while/lstm_cell_6/MatMul/ReadVariableOp*^while/lstm_cell_6/MatMul_1/ReadVariableOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿn2
while/Identity_4
while/Identity_5Identitywhile/lstm_cell_6/add_1:z:0)^while/lstm_cell_6/BiasAdd/ReadVariableOp(^while/lstm_cell_6/MatMul/ReadVariableOp*^while/lstm_cell_6/MatMul_1/ReadVariableOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿn2
while/Identity_5")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"h
1while_lstm_cell_6_biasadd_readvariableop_resource3while_lstm_cell_6_biasadd_readvariableop_resource_0"j
2while_lstm_cell_6_matmul_1_readvariableop_resource4while_lstm_cell_6_matmul_1_readvariableop_resource_0"f
0while_lstm_cell_6_matmul_readvariableop_resource2while_lstm_cell_6_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*Q
_input_shapes@
>: : : : :ÿÿÿÿÿÿÿÿÿn:ÿÿÿÿÿÿÿÿÿn: : :::2T
(while/lstm_cell_6/BiasAdd/ReadVariableOp(while/lstm_cell_6/BiasAdd/ReadVariableOp2R
'while/lstm_cell_6/MatMul/ReadVariableOp'while/lstm_cell_6/MatMul/ReadVariableOp2V
)while/lstm_cell_6/MatMul_1/ReadVariableOp)while/lstm_cell_6/MatMul_1/ReadVariableOp: 
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
:ÿÿÿÿÿÿÿÿÿn:-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿn:

_output_shapes
: :

_output_shapes
: 
B
ô
while_body_63171
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_06
2while_lstm_cell_6_matmul_readvariableop_resource_08
4while_lstm_cell_6_matmul_1_readvariableop_resource_07
3while_lstm_cell_6_biasadd_readvariableop_resource_0
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor4
0while_lstm_cell_6_matmul_readvariableop_resource6
2while_lstm_cell_6_matmul_1_readvariableop_resource5
1while_lstm_cell_6_biasadd_readvariableop_resource¢(while/lstm_cell_6/BiasAdd/ReadVariableOp¢'while/lstm_cell_6/MatMul/ReadVariableOp¢)while/lstm_cell_6/MatMul_1/ReadVariableOpÃ
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
)while/TensorArrayV2Read/TensorListGetItemÆ
'while/lstm_cell_6/MatMul/ReadVariableOpReadVariableOp2while_lstm_cell_6_matmul_readvariableop_resource_0*
_output_shapes
:	¸*
dtype02)
'while/lstm_cell_6/MatMul/ReadVariableOpÔ
while/lstm_cell_6/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0/while/lstm_cell_6/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¸2
while/lstm_cell_6/MatMulÌ
)while/lstm_cell_6/MatMul_1/ReadVariableOpReadVariableOp4while_lstm_cell_6_matmul_1_readvariableop_resource_0*
_output_shapes
:	n¸*
dtype02+
)while/lstm_cell_6/MatMul_1/ReadVariableOp½
while/lstm_cell_6/MatMul_1MatMulwhile_placeholder_21while/lstm_cell_6/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¸2
while/lstm_cell_6/MatMul_1´
while/lstm_cell_6/addAddV2"while/lstm_cell_6/MatMul:product:0$while/lstm_cell_6/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¸2
while/lstm_cell_6/addÅ
(while/lstm_cell_6/BiasAdd/ReadVariableOpReadVariableOp3while_lstm_cell_6_biasadd_readvariableop_resource_0*
_output_shapes	
:¸*
dtype02*
(while/lstm_cell_6/BiasAdd/ReadVariableOpÁ
while/lstm_cell_6/BiasAddBiasAddwhile/lstm_cell_6/add:z:00while/lstm_cell_6/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¸2
while/lstm_cell_6/BiasAddt
while/lstm_cell_6/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
while/lstm_cell_6/Const
!while/lstm_cell_6/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2#
!while/lstm_cell_6/split/split_dim
while/lstm_cell_6/splitSplit*while/lstm_cell_6/split/split_dim:output:0"while/lstm_cell_6/BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿn:ÿÿÿÿÿÿÿÿÿn:ÿÿÿÿÿÿÿÿÿn:ÿÿÿÿÿÿÿÿÿn*
	num_split2
while/lstm_cell_6/split
while/lstm_cell_6/SigmoidSigmoid while/lstm_cell_6/split:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿn2
while/lstm_cell_6/Sigmoid
while/lstm_cell_6/Sigmoid_1Sigmoid while/lstm_cell_6/split:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿn2
while/lstm_cell_6/Sigmoid_1
while/lstm_cell_6/mulMulwhile/lstm_cell_6/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿn2
while/lstm_cell_6/mul
while/lstm_cell_6/ReluRelu while/lstm_cell_6/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿn2
while/lstm_cell_6/Relu°
while/lstm_cell_6/mul_1Mulwhile/lstm_cell_6/Sigmoid:y:0$while/lstm_cell_6/Relu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿn2
while/lstm_cell_6/mul_1¥
while/lstm_cell_6/add_1AddV2while/lstm_cell_6/mul:z:0while/lstm_cell_6/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿn2
while/lstm_cell_6/add_1
while/lstm_cell_6/Sigmoid_2Sigmoid while/lstm_cell_6/split:output:3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿn2
while/lstm_cell_6/Sigmoid_2
while/lstm_cell_6/Relu_1Reluwhile/lstm_cell_6/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿn2
while/lstm_cell_6/Relu_1´
while/lstm_cell_6/mul_2Mulwhile/lstm_cell_6/Sigmoid_2:y:0&while/lstm_cell_6/Relu_1:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿn2
while/lstm_cell_6/mul_2ß
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_6/mul_2:z:0*
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
while/add_1ß
while/IdentityIdentitywhile/add_1:z:0)^while/lstm_cell_6/BiasAdd/ReadVariableOp(^while/lstm_cell_6/MatMul/ReadVariableOp*^while/lstm_cell_6/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identityò
while/Identity_1Identitywhile_while_maximum_iterations)^while/lstm_cell_6/BiasAdd/ReadVariableOp(^while/lstm_cell_6/MatMul/ReadVariableOp*^while/lstm_cell_6/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_1á
while/Identity_2Identitywhile/add:z:0)^while/lstm_cell_6/BiasAdd/ReadVariableOp(^while/lstm_cell_6/MatMul/ReadVariableOp*^while/lstm_cell_6/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_2
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0)^while/lstm_cell_6/BiasAdd/ReadVariableOp(^while/lstm_cell_6/MatMul/ReadVariableOp*^while/lstm_cell_6/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_3
while/Identity_4Identitywhile/lstm_cell_6/mul_2:z:0)^while/lstm_cell_6/BiasAdd/ReadVariableOp(^while/lstm_cell_6/MatMul/ReadVariableOp*^while/lstm_cell_6/MatMul_1/ReadVariableOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿn2
while/Identity_4
while/Identity_5Identitywhile/lstm_cell_6/add_1:z:0)^while/lstm_cell_6/BiasAdd/ReadVariableOp(^while/lstm_cell_6/MatMul/ReadVariableOp*^while/lstm_cell_6/MatMul_1/ReadVariableOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿn2
while/Identity_5")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"h
1while_lstm_cell_6_biasadd_readvariableop_resource3while_lstm_cell_6_biasadd_readvariableop_resource_0"j
2while_lstm_cell_6_matmul_1_readvariableop_resource4while_lstm_cell_6_matmul_1_readvariableop_resource_0"f
0while_lstm_cell_6_matmul_readvariableop_resource2while_lstm_cell_6_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*Q
_input_shapes@
>: : : : :ÿÿÿÿÿÿÿÿÿn:ÿÿÿÿÿÿÿÿÿn: : :::2T
(while/lstm_cell_6/BiasAdd/ReadVariableOp(while/lstm_cell_6/BiasAdd/ReadVariableOp2R
'while/lstm_cell_6/MatMul/ReadVariableOp'while/lstm_cell_6/MatMul/ReadVariableOp2V
)while/lstm_cell_6/MatMul_1/ReadVariableOp)while/lstm_cell_6/MatMul_1/ReadVariableOp: 
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
:ÿÿÿÿÿÿÿÿÿn:-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿn:

_output_shapes
: :

_output_shapes
: 


&__inference_lstm_6_layer_call_fn_63103
inputs_0
unknown
	unknown_0
	unknown_1
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputs_0unknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿn*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *J
fERC
A__inference_lstm_6_layer_call_and_return_conditional_losses_619482
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿn2

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
ÚL
Ô	
lstm_6_while_body_62495*
&lstm_6_while_lstm_6_while_loop_counter0
,lstm_6_while_lstm_6_while_maximum_iterations
lstm_6_while_placeholder
lstm_6_while_placeholder_1
lstm_6_while_placeholder_2
lstm_6_while_placeholder_3)
%lstm_6_while_lstm_6_strided_slice_1_0e
alstm_6_while_tensorarrayv2read_tensorlistgetitem_lstm_6_tensorarrayunstack_tensorlistfromtensor_0=
9lstm_6_while_lstm_cell_6_matmul_readvariableop_resource_0?
;lstm_6_while_lstm_cell_6_matmul_1_readvariableop_resource_0>
:lstm_6_while_lstm_cell_6_biasadd_readvariableop_resource_0
lstm_6_while_identity
lstm_6_while_identity_1
lstm_6_while_identity_2
lstm_6_while_identity_3
lstm_6_while_identity_4
lstm_6_while_identity_5'
#lstm_6_while_lstm_6_strided_slice_1c
_lstm_6_while_tensorarrayv2read_tensorlistgetitem_lstm_6_tensorarrayunstack_tensorlistfromtensor;
7lstm_6_while_lstm_cell_6_matmul_readvariableop_resource=
9lstm_6_while_lstm_cell_6_matmul_1_readvariableop_resource<
8lstm_6_while_lstm_cell_6_biasadd_readvariableop_resource¢/lstm_6/while/lstm_cell_6/BiasAdd/ReadVariableOp¢.lstm_6/while/lstm_cell_6/MatMul/ReadVariableOp¢0lstm_6/while/lstm_cell_6/MatMul_1/ReadVariableOpÑ
>lstm_6/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   2@
>lstm_6/while/TensorArrayV2Read/TensorListGetItem/element_shapeý
0lstm_6/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemalstm_6_while_tensorarrayv2read_tensorlistgetitem_lstm_6_tensorarrayunstack_tensorlistfromtensor_0lstm_6_while_placeholderGlstm_6/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
element_dtype022
0lstm_6/while/TensorArrayV2Read/TensorListGetItemÛ
.lstm_6/while/lstm_cell_6/MatMul/ReadVariableOpReadVariableOp9lstm_6_while_lstm_cell_6_matmul_readvariableop_resource_0*
_output_shapes
:	¸*
dtype020
.lstm_6/while/lstm_cell_6/MatMul/ReadVariableOpð
lstm_6/while/lstm_cell_6/MatMulMatMul7lstm_6/while/TensorArrayV2Read/TensorListGetItem:item:06lstm_6/while/lstm_cell_6/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¸2!
lstm_6/while/lstm_cell_6/MatMulá
0lstm_6/while/lstm_cell_6/MatMul_1/ReadVariableOpReadVariableOp;lstm_6_while_lstm_cell_6_matmul_1_readvariableop_resource_0*
_output_shapes
:	n¸*
dtype022
0lstm_6/while/lstm_cell_6/MatMul_1/ReadVariableOpÙ
!lstm_6/while/lstm_cell_6/MatMul_1MatMullstm_6_while_placeholder_28lstm_6/while/lstm_cell_6/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¸2#
!lstm_6/while/lstm_cell_6/MatMul_1Ð
lstm_6/while/lstm_cell_6/addAddV2)lstm_6/while/lstm_cell_6/MatMul:product:0+lstm_6/while/lstm_cell_6/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¸2
lstm_6/while/lstm_cell_6/addÚ
/lstm_6/while/lstm_cell_6/BiasAdd/ReadVariableOpReadVariableOp:lstm_6_while_lstm_cell_6_biasadd_readvariableop_resource_0*
_output_shapes	
:¸*
dtype021
/lstm_6/while/lstm_cell_6/BiasAdd/ReadVariableOpÝ
 lstm_6/while/lstm_cell_6/BiasAddBiasAdd lstm_6/while/lstm_cell_6/add:z:07lstm_6/while/lstm_cell_6/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¸2"
 lstm_6/while/lstm_cell_6/BiasAdd
lstm_6/while/lstm_cell_6/ConstConst*
_output_shapes
: *
dtype0*
value	B :2 
lstm_6/while/lstm_cell_6/Const
(lstm_6/while/lstm_cell_6/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2*
(lstm_6/while/lstm_cell_6/split/split_dim£
lstm_6/while/lstm_cell_6/splitSplit1lstm_6/while/lstm_cell_6/split/split_dim:output:0)lstm_6/while/lstm_cell_6/BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿn:ÿÿÿÿÿÿÿÿÿn:ÿÿÿÿÿÿÿÿÿn:ÿÿÿÿÿÿÿÿÿn*
	num_split2 
lstm_6/while/lstm_cell_6/splitª
 lstm_6/while/lstm_cell_6/SigmoidSigmoid'lstm_6/while/lstm_cell_6/split:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿn2"
 lstm_6/while/lstm_cell_6/Sigmoid®
"lstm_6/while/lstm_cell_6/Sigmoid_1Sigmoid'lstm_6/while/lstm_cell_6/split:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿn2$
"lstm_6/while/lstm_cell_6/Sigmoid_1¹
lstm_6/while/lstm_cell_6/mulMul&lstm_6/while/lstm_cell_6/Sigmoid_1:y:0lstm_6_while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿn2
lstm_6/while/lstm_cell_6/mul¡
lstm_6/while/lstm_cell_6/ReluRelu'lstm_6/while/lstm_cell_6/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿn2
lstm_6/while/lstm_cell_6/ReluÌ
lstm_6/while/lstm_cell_6/mul_1Mul$lstm_6/while/lstm_cell_6/Sigmoid:y:0+lstm_6/while/lstm_cell_6/Relu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿn2 
lstm_6/while/lstm_cell_6/mul_1Á
lstm_6/while/lstm_cell_6/add_1AddV2 lstm_6/while/lstm_cell_6/mul:z:0"lstm_6/while/lstm_cell_6/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿn2 
lstm_6/while/lstm_cell_6/add_1®
"lstm_6/while/lstm_cell_6/Sigmoid_2Sigmoid'lstm_6/while/lstm_cell_6/split:output:3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿn2$
"lstm_6/while/lstm_cell_6/Sigmoid_2 
lstm_6/while/lstm_cell_6/Relu_1Relu"lstm_6/while/lstm_cell_6/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿn2!
lstm_6/while/lstm_cell_6/Relu_1Ð
lstm_6/while/lstm_cell_6/mul_2Mul&lstm_6/while/lstm_cell_6/Sigmoid_2:y:0-lstm_6/while/lstm_cell_6/Relu_1:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿn2 
lstm_6/while/lstm_cell_6/mul_2
1lstm_6/while/TensorArrayV2Write/TensorListSetItemTensorListSetItemlstm_6_while_placeholder_1lstm_6_while_placeholder"lstm_6/while/lstm_cell_6/mul_2:z:0*
_output_shapes
: *
element_dtype023
1lstm_6/while/TensorArrayV2Write/TensorListSetItemj
lstm_6/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_6/while/add/y
lstm_6/while/addAddV2lstm_6_while_placeholderlstm_6/while/add/y:output:0*
T0*
_output_shapes
: 2
lstm_6/while/addn
lstm_6/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_6/while/add_1/y
lstm_6/while/add_1AddV2&lstm_6_while_lstm_6_while_loop_counterlstm_6/while/add_1/y:output:0*
T0*
_output_shapes
: 2
lstm_6/while/add_1
lstm_6/while/IdentityIdentitylstm_6/while/add_1:z:00^lstm_6/while/lstm_cell_6/BiasAdd/ReadVariableOp/^lstm_6/while/lstm_cell_6/MatMul/ReadVariableOp1^lstm_6/while/lstm_cell_6/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
lstm_6/while/Identity£
lstm_6/while/Identity_1Identity,lstm_6_while_lstm_6_while_maximum_iterations0^lstm_6/while/lstm_cell_6/BiasAdd/ReadVariableOp/^lstm_6/while/lstm_cell_6/MatMul/ReadVariableOp1^lstm_6/while/lstm_cell_6/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
lstm_6/while/Identity_1
lstm_6/while/Identity_2Identitylstm_6/while/add:z:00^lstm_6/while/lstm_cell_6/BiasAdd/ReadVariableOp/^lstm_6/while/lstm_cell_6/MatMul/ReadVariableOp1^lstm_6/while/lstm_cell_6/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
lstm_6/while/Identity_2¸
lstm_6/while/Identity_3IdentityAlstm_6/while/TensorArrayV2Write/TensorListSetItem:output_handle:00^lstm_6/while/lstm_cell_6/BiasAdd/ReadVariableOp/^lstm_6/while/lstm_cell_6/MatMul/ReadVariableOp1^lstm_6/while/lstm_cell_6/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
lstm_6/while/Identity_3ª
lstm_6/while/Identity_4Identity"lstm_6/while/lstm_cell_6/mul_2:z:00^lstm_6/while/lstm_cell_6/BiasAdd/ReadVariableOp/^lstm_6/while/lstm_cell_6/MatMul/ReadVariableOp1^lstm_6/while/lstm_cell_6/MatMul_1/ReadVariableOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿn2
lstm_6/while/Identity_4ª
lstm_6/while/Identity_5Identity"lstm_6/while/lstm_cell_6/add_1:z:00^lstm_6/while/lstm_cell_6/BiasAdd/ReadVariableOp/^lstm_6/while/lstm_cell_6/MatMul/ReadVariableOp1^lstm_6/while/lstm_cell_6/MatMul_1/ReadVariableOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿn2
lstm_6/while/Identity_5"7
lstm_6_while_identitylstm_6/while/Identity:output:0";
lstm_6_while_identity_1 lstm_6/while/Identity_1:output:0";
lstm_6_while_identity_2 lstm_6/while/Identity_2:output:0";
lstm_6_while_identity_3 lstm_6/while/Identity_3:output:0";
lstm_6_while_identity_4 lstm_6/while/Identity_4:output:0";
lstm_6_while_identity_5 lstm_6/while/Identity_5:output:0"L
#lstm_6_while_lstm_6_strided_slice_1%lstm_6_while_lstm_6_strided_slice_1_0"v
8lstm_6_while_lstm_cell_6_biasadd_readvariableop_resource:lstm_6_while_lstm_cell_6_biasadd_readvariableop_resource_0"x
9lstm_6_while_lstm_cell_6_matmul_1_readvariableop_resource;lstm_6_while_lstm_cell_6_matmul_1_readvariableop_resource_0"t
7lstm_6_while_lstm_cell_6_matmul_readvariableop_resource9lstm_6_while_lstm_cell_6_matmul_readvariableop_resource_0"Ä
_lstm_6_while_tensorarrayv2read_tensorlistgetitem_lstm_6_tensorarrayunstack_tensorlistfromtensoralstm_6_while_tensorarrayv2read_tensorlistgetitem_lstm_6_tensorarrayunstack_tensorlistfromtensor_0*Q
_input_shapes@
>: : : : :ÿÿÿÿÿÿÿÿÿn:ÿÿÿÿÿÿÿÿÿn: : :::2b
/lstm_6/while/lstm_cell_6/BiasAdd/ReadVariableOp/lstm_6/while/lstm_cell_6/BiasAdd/ReadVariableOp2`
.lstm_6/while/lstm_cell_6/MatMul/ReadVariableOp.lstm_6/while/lstm_cell_6/MatMul/ReadVariableOp2d
0lstm_6/while/lstm_cell_6/MatMul_1/ReadVariableOp0lstm_6/while/lstm_cell_6/MatMul_1/ReadVariableOp: 
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
:ÿÿÿÿÿÿÿÿÿn:-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿn:

_output_shapes
: :

_output_shapes
: 
½D
Ó
A__inference_lstm_6_layer_call_and_return_conditional_losses_61948

inputs
lstm_cell_6_61866
lstm_cell_6_61868
lstm_cell_6_61870
identity¢#lstm_cell_6/StatefulPartitionedCall¢whileD
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
value	B :n2
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
value	B :n2
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
:ÿÿÿÿÿÿÿÿÿn2
zeros`
zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B :n2
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
value	B :n2
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
:ÿÿÿÿÿÿÿÿÿn2	
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
strided_slice_2
#lstm_cell_6/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_2:output:0zeros:output:0zeros_1:output:0lstm_cell_6_61866lstm_cell_6_61868lstm_cell_6_61870*
Tin

2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿn:ÿÿÿÿÿÿÿÿÿn:ÿÿÿÿÿÿÿÿÿn*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_lstm_cell_6_layer_call_and_return_conditional_losses_614532%
#lstm_cell_6/StatefulPartitionedCall
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿn   2
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
while/loop_counter
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0lstm_cell_6_61866lstm_cell_6_61868lstm_cell_6_61870*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :ÿÿÿÿÿÿÿÿÿn:ÿÿÿÿÿÿÿÿÿn: : : : : *%
_read_only_resource_inputs
	
*
bodyR
while_body_61879*
condR
while_cond_61878*K
output_shapes:
8: : : : :ÿÿÿÿÿÿÿÿÿn:ÿÿÿÿÿÿÿÿÿn: : : : : *
parallel_iterations 2
whileµ
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿn   22
0TensorArrayV2Stack/TensorListStack/element_shapeñ
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿn*
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
:ÿÿÿÿÿÿÿÿÿn*
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
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿn2
transpose_1f
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2	
runtime
IdentityIdentitystrided_slice_3:output:0$^lstm_cell_6/StatefulPartitionedCall^while*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿn2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:::2J
#lstm_cell_6/StatefulPartitionedCall#lstm_cell_6/StatefulPartitionedCall2
whilewhile:\ X
4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
½D
Ó
A__inference_lstm_6_layer_call_and_return_conditional_losses_61816

inputs
lstm_cell_6_61734
lstm_cell_6_61736
lstm_cell_6_61738
identity¢#lstm_cell_6/StatefulPartitionedCall¢whileD
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
value	B :n2
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
value	B :n2
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
:ÿÿÿÿÿÿÿÿÿn2
zeros`
zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B :n2
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
value	B :n2
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
:ÿÿÿÿÿÿÿÿÿn2	
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
strided_slice_2
#lstm_cell_6/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_2:output:0zeros:output:0zeros_1:output:0lstm_cell_6_61734lstm_cell_6_61736lstm_cell_6_61738*
Tin

2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿn:ÿÿÿÿÿÿÿÿÿn:ÿÿÿÿÿÿÿÿÿn*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_lstm_cell_6_layer_call_and_return_conditional_losses_614202%
#lstm_cell_6/StatefulPartitionedCall
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿn   2
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
while/loop_counter
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0lstm_cell_6_61734lstm_cell_6_61736lstm_cell_6_61738*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :ÿÿÿÿÿÿÿÿÿn:ÿÿÿÿÿÿÿÿÿn: : : : : *%
_read_only_resource_inputs
	
*
bodyR
while_body_61747*
condR
while_cond_61746*K
output_shapes:
8: : : : :ÿÿÿÿÿÿÿÿÿn:ÿÿÿÿÿÿÿÿÿn: : : : : *
parallel_iterations 2
whileµ
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿn   22
0TensorArrayV2Stack/TensorListStack/element_shapeñ
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿn*
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
:ÿÿÿÿÿÿÿÿÿn*
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
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿn2
transpose_1f
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2	
runtime
IdentityIdentitystrided_slice_3:output:0$^lstm_cell_6/StatefulPartitionedCall^while*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿn2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:::2J
#lstm_cell_6/StatefulPartitionedCall#lstm_cell_6/StatefulPartitionedCall2
whilewhile:\ X
4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
B
ô
while_body_63324
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_06
2while_lstm_cell_6_matmul_readvariableop_resource_08
4while_lstm_cell_6_matmul_1_readvariableop_resource_07
3while_lstm_cell_6_biasadd_readvariableop_resource_0
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor4
0while_lstm_cell_6_matmul_readvariableop_resource6
2while_lstm_cell_6_matmul_1_readvariableop_resource5
1while_lstm_cell_6_biasadd_readvariableop_resource¢(while/lstm_cell_6/BiasAdd/ReadVariableOp¢'while/lstm_cell_6/MatMul/ReadVariableOp¢)while/lstm_cell_6/MatMul_1/ReadVariableOpÃ
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
)while/TensorArrayV2Read/TensorListGetItemÆ
'while/lstm_cell_6/MatMul/ReadVariableOpReadVariableOp2while_lstm_cell_6_matmul_readvariableop_resource_0*
_output_shapes
:	¸*
dtype02)
'while/lstm_cell_6/MatMul/ReadVariableOpÔ
while/lstm_cell_6/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0/while/lstm_cell_6/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¸2
while/lstm_cell_6/MatMulÌ
)while/lstm_cell_6/MatMul_1/ReadVariableOpReadVariableOp4while_lstm_cell_6_matmul_1_readvariableop_resource_0*
_output_shapes
:	n¸*
dtype02+
)while/lstm_cell_6/MatMul_1/ReadVariableOp½
while/lstm_cell_6/MatMul_1MatMulwhile_placeholder_21while/lstm_cell_6/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¸2
while/lstm_cell_6/MatMul_1´
while/lstm_cell_6/addAddV2"while/lstm_cell_6/MatMul:product:0$while/lstm_cell_6/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¸2
while/lstm_cell_6/addÅ
(while/lstm_cell_6/BiasAdd/ReadVariableOpReadVariableOp3while_lstm_cell_6_biasadd_readvariableop_resource_0*
_output_shapes	
:¸*
dtype02*
(while/lstm_cell_6/BiasAdd/ReadVariableOpÁ
while/lstm_cell_6/BiasAddBiasAddwhile/lstm_cell_6/add:z:00while/lstm_cell_6/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¸2
while/lstm_cell_6/BiasAddt
while/lstm_cell_6/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
while/lstm_cell_6/Const
!while/lstm_cell_6/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2#
!while/lstm_cell_6/split/split_dim
while/lstm_cell_6/splitSplit*while/lstm_cell_6/split/split_dim:output:0"while/lstm_cell_6/BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿn:ÿÿÿÿÿÿÿÿÿn:ÿÿÿÿÿÿÿÿÿn:ÿÿÿÿÿÿÿÿÿn*
	num_split2
while/lstm_cell_6/split
while/lstm_cell_6/SigmoidSigmoid while/lstm_cell_6/split:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿn2
while/lstm_cell_6/Sigmoid
while/lstm_cell_6/Sigmoid_1Sigmoid while/lstm_cell_6/split:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿn2
while/lstm_cell_6/Sigmoid_1
while/lstm_cell_6/mulMulwhile/lstm_cell_6/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿn2
while/lstm_cell_6/mul
while/lstm_cell_6/ReluRelu while/lstm_cell_6/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿn2
while/lstm_cell_6/Relu°
while/lstm_cell_6/mul_1Mulwhile/lstm_cell_6/Sigmoid:y:0$while/lstm_cell_6/Relu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿn2
while/lstm_cell_6/mul_1¥
while/lstm_cell_6/add_1AddV2while/lstm_cell_6/mul:z:0while/lstm_cell_6/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿn2
while/lstm_cell_6/add_1
while/lstm_cell_6/Sigmoid_2Sigmoid while/lstm_cell_6/split:output:3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿn2
while/lstm_cell_6/Sigmoid_2
while/lstm_cell_6/Relu_1Reluwhile/lstm_cell_6/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿn2
while/lstm_cell_6/Relu_1´
while/lstm_cell_6/mul_2Mulwhile/lstm_cell_6/Sigmoid_2:y:0&while/lstm_cell_6/Relu_1:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿn2
while/lstm_cell_6/mul_2ß
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_6/mul_2:z:0*
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
while/add_1ß
while/IdentityIdentitywhile/add_1:z:0)^while/lstm_cell_6/BiasAdd/ReadVariableOp(^while/lstm_cell_6/MatMul/ReadVariableOp*^while/lstm_cell_6/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identityò
while/Identity_1Identitywhile_while_maximum_iterations)^while/lstm_cell_6/BiasAdd/ReadVariableOp(^while/lstm_cell_6/MatMul/ReadVariableOp*^while/lstm_cell_6/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_1á
while/Identity_2Identitywhile/add:z:0)^while/lstm_cell_6/BiasAdd/ReadVariableOp(^while/lstm_cell_6/MatMul/ReadVariableOp*^while/lstm_cell_6/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_2
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0)^while/lstm_cell_6/BiasAdd/ReadVariableOp(^while/lstm_cell_6/MatMul/ReadVariableOp*^while/lstm_cell_6/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_3
while/Identity_4Identitywhile/lstm_cell_6/mul_2:z:0)^while/lstm_cell_6/BiasAdd/ReadVariableOp(^while/lstm_cell_6/MatMul/ReadVariableOp*^while/lstm_cell_6/MatMul_1/ReadVariableOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿn2
while/Identity_4
while/Identity_5Identitywhile/lstm_cell_6/add_1:z:0)^while/lstm_cell_6/BiasAdd/ReadVariableOp(^while/lstm_cell_6/MatMul/ReadVariableOp*^while/lstm_cell_6/MatMul_1/ReadVariableOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿn2
while/Identity_5")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"h
1while_lstm_cell_6_biasadd_readvariableop_resource3while_lstm_cell_6_biasadd_readvariableop_resource_0"j
2while_lstm_cell_6_matmul_1_readvariableop_resource4while_lstm_cell_6_matmul_1_readvariableop_resource_0"f
0while_lstm_cell_6_matmul_readvariableop_resource2while_lstm_cell_6_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*Q
_input_shapes@
>: : : : :ÿÿÿÿÿÿÿÿÿn:ÿÿÿÿÿÿÿÿÿn: : :::2T
(while/lstm_cell_6/BiasAdd/ReadVariableOp(while/lstm_cell_6/BiasAdd/ReadVariableOp2R
'while/lstm_cell_6/MatMul/ReadVariableOp'while/lstm_cell_6/MatMul/ReadVariableOp2V
)while/lstm_cell_6/MatMul_1/ReadVariableOp)while/lstm_cell_6/MatMul_1/ReadVariableOp: 
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
:ÿÿÿÿÿÿÿÿÿn:-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿn:

_output_shapes
: :

_output_shapes
: 
`
ô
$sequential_6_lstm_6_while_body_61256D
@sequential_6_lstm_6_while_sequential_6_lstm_6_while_loop_counterJ
Fsequential_6_lstm_6_while_sequential_6_lstm_6_while_maximum_iterations)
%sequential_6_lstm_6_while_placeholder+
'sequential_6_lstm_6_while_placeholder_1+
'sequential_6_lstm_6_while_placeholder_2+
'sequential_6_lstm_6_while_placeholder_3C
?sequential_6_lstm_6_while_sequential_6_lstm_6_strided_slice_1_0
{sequential_6_lstm_6_while_tensorarrayv2read_tensorlistgetitem_sequential_6_lstm_6_tensorarrayunstack_tensorlistfromtensor_0J
Fsequential_6_lstm_6_while_lstm_cell_6_matmul_readvariableop_resource_0L
Hsequential_6_lstm_6_while_lstm_cell_6_matmul_1_readvariableop_resource_0K
Gsequential_6_lstm_6_while_lstm_cell_6_biasadd_readvariableop_resource_0&
"sequential_6_lstm_6_while_identity(
$sequential_6_lstm_6_while_identity_1(
$sequential_6_lstm_6_while_identity_2(
$sequential_6_lstm_6_while_identity_3(
$sequential_6_lstm_6_while_identity_4(
$sequential_6_lstm_6_while_identity_5A
=sequential_6_lstm_6_while_sequential_6_lstm_6_strided_slice_1}
ysequential_6_lstm_6_while_tensorarrayv2read_tensorlistgetitem_sequential_6_lstm_6_tensorarrayunstack_tensorlistfromtensorH
Dsequential_6_lstm_6_while_lstm_cell_6_matmul_readvariableop_resourceJ
Fsequential_6_lstm_6_while_lstm_cell_6_matmul_1_readvariableop_resourceI
Esequential_6_lstm_6_while_lstm_cell_6_biasadd_readvariableop_resource¢<sequential_6/lstm_6/while/lstm_cell_6/BiasAdd/ReadVariableOp¢;sequential_6/lstm_6/while/lstm_cell_6/MatMul/ReadVariableOp¢=sequential_6/lstm_6/while/lstm_cell_6/MatMul_1/ReadVariableOpë
Ksequential_6/lstm_6/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   2M
Ksequential_6/lstm_6/while/TensorArrayV2Read/TensorListGetItem/element_shapeË
=sequential_6/lstm_6/while/TensorArrayV2Read/TensorListGetItemTensorListGetItem{sequential_6_lstm_6_while_tensorarrayv2read_tensorlistgetitem_sequential_6_lstm_6_tensorarrayunstack_tensorlistfromtensor_0%sequential_6_lstm_6_while_placeholderTsequential_6/lstm_6/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
element_dtype02?
=sequential_6/lstm_6/while/TensorArrayV2Read/TensorListGetItem
;sequential_6/lstm_6/while/lstm_cell_6/MatMul/ReadVariableOpReadVariableOpFsequential_6_lstm_6_while_lstm_cell_6_matmul_readvariableop_resource_0*
_output_shapes
:	¸*
dtype02=
;sequential_6/lstm_6/while/lstm_cell_6/MatMul/ReadVariableOp¤
,sequential_6/lstm_6/while/lstm_cell_6/MatMulMatMulDsequential_6/lstm_6/while/TensorArrayV2Read/TensorListGetItem:item:0Csequential_6/lstm_6/while/lstm_cell_6/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¸2.
,sequential_6/lstm_6/while/lstm_cell_6/MatMul
=sequential_6/lstm_6/while/lstm_cell_6/MatMul_1/ReadVariableOpReadVariableOpHsequential_6_lstm_6_while_lstm_cell_6_matmul_1_readvariableop_resource_0*
_output_shapes
:	n¸*
dtype02?
=sequential_6/lstm_6/while/lstm_cell_6/MatMul_1/ReadVariableOp
.sequential_6/lstm_6/while/lstm_cell_6/MatMul_1MatMul'sequential_6_lstm_6_while_placeholder_2Esequential_6/lstm_6/while/lstm_cell_6/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¸20
.sequential_6/lstm_6/while/lstm_cell_6/MatMul_1
)sequential_6/lstm_6/while/lstm_cell_6/addAddV26sequential_6/lstm_6/while/lstm_cell_6/MatMul:product:08sequential_6/lstm_6/while/lstm_cell_6/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¸2+
)sequential_6/lstm_6/while/lstm_cell_6/add
<sequential_6/lstm_6/while/lstm_cell_6/BiasAdd/ReadVariableOpReadVariableOpGsequential_6_lstm_6_while_lstm_cell_6_biasadd_readvariableop_resource_0*
_output_shapes	
:¸*
dtype02>
<sequential_6/lstm_6/while/lstm_cell_6/BiasAdd/ReadVariableOp
-sequential_6/lstm_6/while/lstm_cell_6/BiasAddBiasAdd-sequential_6/lstm_6/while/lstm_cell_6/add:z:0Dsequential_6/lstm_6/while/lstm_cell_6/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¸2/
-sequential_6/lstm_6/while/lstm_cell_6/BiasAdd
+sequential_6/lstm_6/while/lstm_cell_6/ConstConst*
_output_shapes
: *
dtype0*
value	B :2-
+sequential_6/lstm_6/while/lstm_cell_6/Const°
5sequential_6/lstm_6/while/lstm_cell_6/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :27
5sequential_6/lstm_6/while/lstm_cell_6/split/split_dim×
+sequential_6/lstm_6/while/lstm_cell_6/splitSplit>sequential_6/lstm_6/while/lstm_cell_6/split/split_dim:output:06sequential_6/lstm_6/while/lstm_cell_6/BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿn:ÿÿÿÿÿÿÿÿÿn:ÿÿÿÿÿÿÿÿÿn:ÿÿÿÿÿÿÿÿÿn*
	num_split2-
+sequential_6/lstm_6/while/lstm_cell_6/splitÑ
-sequential_6/lstm_6/while/lstm_cell_6/SigmoidSigmoid4sequential_6/lstm_6/while/lstm_cell_6/split:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿn2/
-sequential_6/lstm_6/while/lstm_cell_6/SigmoidÕ
/sequential_6/lstm_6/while/lstm_cell_6/Sigmoid_1Sigmoid4sequential_6/lstm_6/while/lstm_cell_6/split:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿn21
/sequential_6/lstm_6/while/lstm_cell_6/Sigmoid_1í
)sequential_6/lstm_6/while/lstm_cell_6/mulMul3sequential_6/lstm_6/while/lstm_cell_6/Sigmoid_1:y:0'sequential_6_lstm_6_while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿn2+
)sequential_6/lstm_6/while/lstm_cell_6/mulÈ
*sequential_6/lstm_6/while/lstm_cell_6/ReluRelu4sequential_6/lstm_6/while/lstm_cell_6/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿn2,
*sequential_6/lstm_6/while/lstm_cell_6/Relu
+sequential_6/lstm_6/while/lstm_cell_6/mul_1Mul1sequential_6/lstm_6/while/lstm_cell_6/Sigmoid:y:08sequential_6/lstm_6/while/lstm_cell_6/Relu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿn2-
+sequential_6/lstm_6/while/lstm_cell_6/mul_1õ
+sequential_6/lstm_6/while/lstm_cell_6/add_1AddV2-sequential_6/lstm_6/while/lstm_cell_6/mul:z:0/sequential_6/lstm_6/while/lstm_cell_6/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿn2-
+sequential_6/lstm_6/while/lstm_cell_6/add_1Õ
/sequential_6/lstm_6/while/lstm_cell_6/Sigmoid_2Sigmoid4sequential_6/lstm_6/while/lstm_cell_6/split:output:3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿn21
/sequential_6/lstm_6/while/lstm_cell_6/Sigmoid_2Ç
,sequential_6/lstm_6/while/lstm_cell_6/Relu_1Relu/sequential_6/lstm_6/while/lstm_cell_6/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿn2.
,sequential_6/lstm_6/while/lstm_cell_6/Relu_1
+sequential_6/lstm_6/while/lstm_cell_6/mul_2Mul3sequential_6/lstm_6/while/lstm_cell_6/Sigmoid_2:y:0:sequential_6/lstm_6/while/lstm_cell_6/Relu_1:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿn2-
+sequential_6/lstm_6/while/lstm_cell_6/mul_2Ã
>sequential_6/lstm_6/while/TensorArrayV2Write/TensorListSetItemTensorListSetItem'sequential_6_lstm_6_while_placeholder_1%sequential_6_lstm_6_while_placeholder/sequential_6/lstm_6/while/lstm_cell_6/mul_2:z:0*
_output_shapes
: *
element_dtype02@
>sequential_6/lstm_6/while/TensorArrayV2Write/TensorListSetItem
sequential_6/lstm_6/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2!
sequential_6/lstm_6/while/add/y¹
sequential_6/lstm_6/while/addAddV2%sequential_6_lstm_6_while_placeholder(sequential_6/lstm_6/while/add/y:output:0*
T0*
_output_shapes
: 2
sequential_6/lstm_6/while/add
!sequential_6/lstm_6/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2#
!sequential_6/lstm_6/while/add_1/yÚ
sequential_6/lstm_6/while/add_1AddV2@sequential_6_lstm_6_while_sequential_6_lstm_6_while_loop_counter*sequential_6/lstm_6/while/add_1/y:output:0*
T0*
_output_shapes
: 2!
sequential_6/lstm_6/while/add_1×
"sequential_6/lstm_6/while/IdentityIdentity#sequential_6/lstm_6/while/add_1:z:0=^sequential_6/lstm_6/while/lstm_cell_6/BiasAdd/ReadVariableOp<^sequential_6/lstm_6/while/lstm_cell_6/MatMul/ReadVariableOp>^sequential_6/lstm_6/while/lstm_cell_6/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2$
"sequential_6/lstm_6/while/Identityþ
$sequential_6/lstm_6/while/Identity_1IdentityFsequential_6_lstm_6_while_sequential_6_lstm_6_while_maximum_iterations=^sequential_6/lstm_6/while/lstm_cell_6/BiasAdd/ReadVariableOp<^sequential_6/lstm_6/while/lstm_cell_6/MatMul/ReadVariableOp>^sequential_6/lstm_6/while/lstm_cell_6/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2&
$sequential_6/lstm_6/while/Identity_1Ù
$sequential_6/lstm_6/while/Identity_2Identity!sequential_6/lstm_6/while/add:z:0=^sequential_6/lstm_6/while/lstm_cell_6/BiasAdd/ReadVariableOp<^sequential_6/lstm_6/while/lstm_cell_6/MatMul/ReadVariableOp>^sequential_6/lstm_6/while/lstm_cell_6/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2&
$sequential_6/lstm_6/while/Identity_2
$sequential_6/lstm_6/while/Identity_3IdentityNsequential_6/lstm_6/while/TensorArrayV2Write/TensorListSetItem:output_handle:0=^sequential_6/lstm_6/while/lstm_cell_6/BiasAdd/ReadVariableOp<^sequential_6/lstm_6/while/lstm_cell_6/MatMul/ReadVariableOp>^sequential_6/lstm_6/while/lstm_cell_6/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2&
$sequential_6/lstm_6/while/Identity_3ø
$sequential_6/lstm_6/while/Identity_4Identity/sequential_6/lstm_6/while/lstm_cell_6/mul_2:z:0=^sequential_6/lstm_6/while/lstm_cell_6/BiasAdd/ReadVariableOp<^sequential_6/lstm_6/while/lstm_cell_6/MatMul/ReadVariableOp>^sequential_6/lstm_6/while/lstm_cell_6/MatMul_1/ReadVariableOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿn2&
$sequential_6/lstm_6/while/Identity_4ø
$sequential_6/lstm_6/while/Identity_5Identity/sequential_6/lstm_6/while/lstm_cell_6/add_1:z:0=^sequential_6/lstm_6/while/lstm_cell_6/BiasAdd/ReadVariableOp<^sequential_6/lstm_6/while/lstm_cell_6/MatMul/ReadVariableOp>^sequential_6/lstm_6/while/lstm_cell_6/MatMul_1/ReadVariableOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿn2&
$sequential_6/lstm_6/while/Identity_5"Q
"sequential_6_lstm_6_while_identity+sequential_6/lstm_6/while/Identity:output:0"U
$sequential_6_lstm_6_while_identity_1-sequential_6/lstm_6/while/Identity_1:output:0"U
$sequential_6_lstm_6_while_identity_2-sequential_6/lstm_6/while/Identity_2:output:0"U
$sequential_6_lstm_6_while_identity_3-sequential_6/lstm_6/while/Identity_3:output:0"U
$sequential_6_lstm_6_while_identity_4-sequential_6/lstm_6/while/Identity_4:output:0"U
$sequential_6_lstm_6_while_identity_5-sequential_6/lstm_6/while/Identity_5:output:0"
Esequential_6_lstm_6_while_lstm_cell_6_biasadd_readvariableop_resourceGsequential_6_lstm_6_while_lstm_cell_6_biasadd_readvariableop_resource_0"
Fsequential_6_lstm_6_while_lstm_cell_6_matmul_1_readvariableop_resourceHsequential_6_lstm_6_while_lstm_cell_6_matmul_1_readvariableop_resource_0"
Dsequential_6_lstm_6_while_lstm_cell_6_matmul_readvariableop_resourceFsequential_6_lstm_6_while_lstm_cell_6_matmul_readvariableop_resource_0"
=sequential_6_lstm_6_while_sequential_6_lstm_6_strided_slice_1?sequential_6_lstm_6_while_sequential_6_lstm_6_strided_slice_1_0"ø
ysequential_6_lstm_6_while_tensorarrayv2read_tensorlistgetitem_sequential_6_lstm_6_tensorarrayunstack_tensorlistfromtensor{sequential_6_lstm_6_while_tensorarrayv2read_tensorlistgetitem_sequential_6_lstm_6_tensorarrayunstack_tensorlistfromtensor_0*Q
_input_shapes@
>: : : : :ÿÿÿÿÿÿÿÿÿn:ÿÿÿÿÿÿÿÿÿn: : :::2|
<sequential_6/lstm_6/while/lstm_cell_6/BiasAdd/ReadVariableOp<sequential_6/lstm_6/while/lstm_cell_6/BiasAdd/ReadVariableOp2z
;sequential_6/lstm_6/while/lstm_cell_6/MatMul/ReadVariableOp;sequential_6/lstm_6/while/lstm_cell_6/MatMul/ReadVariableOp2~
=sequential_6/lstm_6/while/lstm_cell_6/MatMul_1/ReadVariableOp=sequential_6/lstm_6/while/lstm_cell_6/MatMul_1/ReadVariableOp: 
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
:ÿÿÿÿÿÿÿÿÿn:-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿn:

_output_shapes
: :

_output_shapes
: 

Î
$sequential_6_lstm_6_while_cond_61255D
@sequential_6_lstm_6_while_sequential_6_lstm_6_while_loop_counterJ
Fsequential_6_lstm_6_while_sequential_6_lstm_6_while_maximum_iterations)
%sequential_6_lstm_6_while_placeholder+
'sequential_6_lstm_6_while_placeholder_1+
'sequential_6_lstm_6_while_placeholder_2+
'sequential_6_lstm_6_while_placeholder_3F
Bsequential_6_lstm_6_while_less_sequential_6_lstm_6_strided_slice_1[
Wsequential_6_lstm_6_while_sequential_6_lstm_6_while_cond_61255___redundant_placeholder0[
Wsequential_6_lstm_6_while_sequential_6_lstm_6_while_cond_61255___redundant_placeholder1[
Wsequential_6_lstm_6_while_sequential_6_lstm_6_while_cond_61255___redundant_placeholder2[
Wsequential_6_lstm_6_while_sequential_6_lstm_6_while_cond_61255___redundant_placeholder3&
"sequential_6_lstm_6_while_identity
Ô
sequential_6/lstm_6/while/LessLess%sequential_6_lstm_6_while_placeholderBsequential_6_lstm_6_while_less_sequential_6_lstm_6_strided_slice_1*
T0*
_output_shapes
: 2 
sequential_6/lstm_6/while/Less
"sequential_6/lstm_6/while/IdentityIdentity"sequential_6/lstm_6/while/Less:z:0*
T0
*
_output_shapes
: 2$
"sequential_6/lstm_6/while/Identity"Q
"sequential_6_lstm_6_while_identity+sequential_6/lstm_6/while/Identity:output:0*S
_input_shapesB
@: : : : :ÿÿÿÿÿÿÿÿÿn:ÿÿÿÿÿÿÿÿÿn: ::::: 
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
:ÿÿÿÿÿÿÿÿÿn:-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿn:

_output_shapes
: :

_output_shapes
:
·
Ë
+__inference_lstm_cell_6_layer_call_fn_63550

inputs
states_0
states_1
unknown
	unknown_0
	unknown_1
identity

identity_1

identity_2¢StatefulPartitionedCallÁ
StatefulPartitionedCallStatefulPartitionedCallinputsstates_0states_1unknown	unknown_0	unknown_1*
Tin

2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿn:ÿÿÿÿÿÿÿÿÿn:ÿÿÿÿÿÿÿÿÿn*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_lstm_cell_6_layer_call_and_return_conditional_losses_614532
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿn2

Identity

Identity_1Identity StatefulPartitionedCall:output:1^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿn2

Identity_1

Identity_2Identity StatefulPartitionedCall:output:2^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿn2

Identity_2"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*X
_input_shapesG
E:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿn:ÿÿÿÿÿÿÿÿÿn:::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs:QM
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿn
"
_user_specified_name
states/0:QM
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿn
"
_user_specified_name
states/1
B
ô
while_body_62181
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_06
2while_lstm_cell_6_matmul_readvariableop_resource_08
4while_lstm_cell_6_matmul_1_readvariableop_resource_07
3while_lstm_cell_6_biasadd_readvariableop_resource_0
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor4
0while_lstm_cell_6_matmul_readvariableop_resource6
2while_lstm_cell_6_matmul_1_readvariableop_resource5
1while_lstm_cell_6_biasadd_readvariableop_resource¢(while/lstm_cell_6/BiasAdd/ReadVariableOp¢'while/lstm_cell_6/MatMul/ReadVariableOp¢)while/lstm_cell_6/MatMul_1/ReadVariableOpÃ
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
)while/TensorArrayV2Read/TensorListGetItemÆ
'while/lstm_cell_6/MatMul/ReadVariableOpReadVariableOp2while_lstm_cell_6_matmul_readvariableop_resource_0*
_output_shapes
:	¸*
dtype02)
'while/lstm_cell_6/MatMul/ReadVariableOpÔ
while/lstm_cell_6/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0/while/lstm_cell_6/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¸2
while/lstm_cell_6/MatMulÌ
)while/lstm_cell_6/MatMul_1/ReadVariableOpReadVariableOp4while_lstm_cell_6_matmul_1_readvariableop_resource_0*
_output_shapes
:	n¸*
dtype02+
)while/lstm_cell_6/MatMul_1/ReadVariableOp½
while/lstm_cell_6/MatMul_1MatMulwhile_placeholder_21while/lstm_cell_6/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¸2
while/lstm_cell_6/MatMul_1´
while/lstm_cell_6/addAddV2"while/lstm_cell_6/MatMul:product:0$while/lstm_cell_6/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¸2
while/lstm_cell_6/addÅ
(while/lstm_cell_6/BiasAdd/ReadVariableOpReadVariableOp3while_lstm_cell_6_biasadd_readvariableop_resource_0*
_output_shapes	
:¸*
dtype02*
(while/lstm_cell_6/BiasAdd/ReadVariableOpÁ
while/lstm_cell_6/BiasAddBiasAddwhile/lstm_cell_6/add:z:00while/lstm_cell_6/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¸2
while/lstm_cell_6/BiasAddt
while/lstm_cell_6/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
while/lstm_cell_6/Const
!while/lstm_cell_6/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2#
!while/lstm_cell_6/split/split_dim
while/lstm_cell_6/splitSplit*while/lstm_cell_6/split/split_dim:output:0"while/lstm_cell_6/BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿn:ÿÿÿÿÿÿÿÿÿn:ÿÿÿÿÿÿÿÿÿn:ÿÿÿÿÿÿÿÿÿn*
	num_split2
while/lstm_cell_6/split
while/lstm_cell_6/SigmoidSigmoid while/lstm_cell_6/split:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿn2
while/lstm_cell_6/Sigmoid
while/lstm_cell_6/Sigmoid_1Sigmoid while/lstm_cell_6/split:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿn2
while/lstm_cell_6/Sigmoid_1
while/lstm_cell_6/mulMulwhile/lstm_cell_6/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿn2
while/lstm_cell_6/mul
while/lstm_cell_6/ReluRelu while/lstm_cell_6/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿn2
while/lstm_cell_6/Relu°
while/lstm_cell_6/mul_1Mulwhile/lstm_cell_6/Sigmoid:y:0$while/lstm_cell_6/Relu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿn2
while/lstm_cell_6/mul_1¥
while/lstm_cell_6/add_1AddV2while/lstm_cell_6/mul:z:0while/lstm_cell_6/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿn2
while/lstm_cell_6/add_1
while/lstm_cell_6/Sigmoid_2Sigmoid while/lstm_cell_6/split:output:3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿn2
while/lstm_cell_6/Sigmoid_2
while/lstm_cell_6/Relu_1Reluwhile/lstm_cell_6/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿn2
while/lstm_cell_6/Relu_1´
while/lstm_cell_6/mul_2Mulwhile/lstm_cell_6/Sigmoid_2:y:0&while/lstm_cell_6/Relu_1:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿn2
while/lstm_cell_6/mul_2ß
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_6/mul_2:z:0*
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
while/add_1ß
while/IdentityIdentitywhile/add_1:z:0)^while/lstm_cell_6/BiasAdd/ReadVariableOp(^while/lstm_cell_6/MatMul/ReadVariableOp*^while/lstm_cell_6/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identityò
while/Identity_1Identitywhile_while_maximum_iterations)^while/lstm_cell_6/BiasAdd/ReadVariableOp(^while/lstm_cell_6/MatMul/ReadVariableOp*^while/lstm_cell_6/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_1á
while/Identity_2Identitywhile/add:z:0)^while/lstm_cell_6/BiasAdd/ReadVariableOp(^while/lstm_cell_6/MatMul/ReadVariableOp*^while/lstm_cell_6/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_2
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0)^while/lstm_cell_6/BiasAdd/ReadVariableOp(^while/lstm_cell_6/MatMul/ReadVariableOp*^while/lstm_cell_6/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_3
while/Identity_4Identitywhile/lstm_cell_6/mul_2:z:0)^while/lstm_cell_6/BiasAdd/ReadVariableOp(^while/lstm_cell_6/MatMul/ReadVariableOp*^while/lstm_cell_6/MatMul_1/ReadVariableOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿn2
while/Identity_4
while/Identity_5Identitywhile/lstm_cell_6/add_1:z:0)^while/lstm_cell_6/BiasAdd/ReadVariableOp(^while/lstm_cell_6/MatMul/ReadVariableOp*^while/lstm_cell_6/MatMul_1/ReadVariableOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿn2
while/Identity_5")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"h
1while_lstm_cell_6_biasadd_readvariableop_resource3while_lstm_cell_6_biasadd_readvariableop_resource_0"j
2while_lstm_cell_6_matmul_1_readvariableop_resource4while_lstm_cell_6_matmul_1_readvariableop_resource_0"f
0while_lstm_cell_6_matmul_readvariableop_resource2while_lstm_cell_6_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*Q
_input_shapes@
>: : : : :ÿÿÿÿÿÿÿÿÿn:ÿÿÿÿÿÿÿÿÿn: : :::2T
(while/lstm_cell_6/BiasAdd/ReadVariableOp(while/lstm_cell_6/BiasAdd/ReadVariableOp2R
'while/lstm_cell_6/MatMul/ReadVariableOp'while/lstm_cell_6/MatMul/ReadVariableOp2V
)while/lstm_cell_6/MatMul_1/ReadVariableOp)while/lstm_cell_6/MatMul_1/ReadVariableOp: 
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
:ÿÿÿÿÿÿÿÿÿn:-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿn:

_output_shapes
: :

_output_shapes
: 
	
Û
B__inference_dense_6_layer_call_and_return_conditional_losses_63441

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:n*
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
:ÿÿÿÿÿÿÿÿÿn::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿn
 
_user_specified_nameinputs
Ý
´
,__inference_sequential_6_layer_call_fn_62402
lstm_6_input
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
identity¢StatefulPartitionedCall¤
StatefulPartitionedCallStatefulPartitionedCalllstm_6_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3*
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
GPU 2J 8 *P
fKRI
G__inference_sequential_6_layer_call_and_return_conditional_losses_623892
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:ÿÿÿÿÿÿÿÿÿ:::::22
StatefulPartitionedCallStatefulPartitionedCall:Y U
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
&
_user_specified_namelstm_6_input
·
Ë
+__inference_lstm_cell_6_layer_call_fn_63533

inputs
states_0
states_1
unknown
	unknown_0
	unknown_1
identity

identity_1

identity_2¢StatefulPartitionedCallÁ
StatefulPartitionedCallStatefulPartitionedCallinputsstates_0states_1unknown	unknown_0	unknown_1*
Tin

2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿn:ÿÿÿÿÿÿÿÿÿn:ÿÿÿÿÿÿÿÿÿn*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_lstm_cell_6_layer_call_and_return_conditional_losses_614202
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿn2

Identity

Identity_1Identity StatefulPartitionedCall:output:1^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿn2

Identity_1

Identity_2Identity StatefulPartitionedCall:output:2^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿn2

Identity_2"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*X
_input_shapesG
E:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿn:ÿÿÿÿÿÿÿÿÿn:::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs:QM
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿn
"
_user_specified_name
states/0:QM
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿn
"
_user_specified_name
states/1
ù	
Ê
lstm_6_while_cond_62494*
&lstm_6_while_lstm_6_while_loop_counter0
,lstm_6_while_lstm_6_while_maximum_iterations
lstm_6_while_placeholder
lstm_6_while_placeholder_1
lstm_6_while_placeholder_2
lstm_6_while_placeholder_3,
(lstm_6_while_less_lstm_6_strided_slice_1A
=lstm_6_while_lstm_6_while_cond_62494___redundant_placeholder0A
=lstm_6_while_lstm_6_while_cond_62494___redundant_placeholder1A
=lstm_6_while_lstm_6_while_cond_62494___redundant_placeholder2A
=lstm_6_while_lstm_6_while_cond_62494___redundant_placeholder3
lstm_6_while_identity

lstm_6/while/LessLesslstm_6_while_placeholder(lstm_6_while_less_lstm_6_strided_slice_1*
T0*
_output_shapes
: 2
lstm_6/while/Lessr
lstm_6/while/IdentityIdentitylstm_6/while/Less:z:0*
T0
*
_output_shapes
: 2
lstm_6/while/Identity"7
lstm_6_while_identitylstm_6/while/Identity:output:0*S
_input_shapesB
@: : : : :ÿÿÿÿÿÿÿÿÿn:ÿÿÿÿÿÿÿÿÿn: ::::: 
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
:ÿÿÿÿÿÿÿÿÿn:-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿn:

_output_shapes
: :

_output_shapes
:
Ë
®
,__inference_sequential_6_layer_call_fn_62775

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
identity¢StatefulPartitionedCall
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
GPU 2J 8 *P
fKRI
G__inference_sequential_6_layer_call_and_return_conditional_losses_623892
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:ÿÿÿÿÿÿÿÿÿ:::::22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
B
ô
while_body_62843
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_06
2while_lstm_cell_6_matmul_readvariableop_resource_08
4while_lstm_cell_6_matmul_1_readvariableop_resource_07
3while_lstm_cell_6_biasadd_readvariableop_resource_0
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor4
0while_lstm_cell_6_matmul_readvariableop_resource6
2while_lstm_cell_6_matmul_1_readvariableop_resource5
1while_lstm_cell_6_biasadd_readvariableop_resource¢(while/lstm_cell_6/BiasAdd/ReadVariableOp¢'while/lstm_cell_6/MatMul/ReadVariableOp¢)while/lstm_cell_6/MatMul_1/ReadVariableOpÃ
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
)while/TensorArrayV2Read/TensorListGetItemÆ
'while/lstm_cell_6/MatMul/ReadVariableOpReadVariableOp2while_lstm_cell_6_matmul_readvariableop_resource_0*
_output_shapes
:	¸*
dtype02)
'while/lstm_cell_6/MatMul/ReadVariableOpÔ
while/lstm_cell_6/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0/while/lstm_cell_6/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¸2
while/lstm_cell_6/MatMulÌ
)while/lstm_cell_6/MatMul_1/ReadVariableOpReadVariableOp4while_lstm_cell_6_matmul_1_readvariableop_resource_0*
_output_shapes
:	n¸*
dtype02+
)while/lstm_cell_6/MatMul_1/ReadVariableOp½
while/lstm_cell_6/MatMul_1MatMulwhile_placeholder_21while/lstm_cell_6/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¸2
while/lstm_cell_6/MatMul_1´
while/lstm_cell_6/addAddV2"while/lstm_cell_6/MatMul:product:0$while/lstm_cell_6/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¸2
while/lstm_cell_6/addÅ
(while/lstm_cell_6/BiasAdd/ReadVariableOpReadVariableOp3while_lstm_cell_6_biasadd_readvariableop_resource_0*
_output_shapes	
:¸*
dtype02*
(while/lstm_cell_6/BiasAdd/ReadVariableOpÁ
while/lstm_cell_6/BiasAddBiasAddwhile/lstm_cell_6/add:z:00while/lstm_cell_6/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¸2
while/lstm_cell_6/BiasAddt
while/lstm_cell_6/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
while/lstm_cell_6/Const
!while/lstm_cell_6/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2#
!while/lstm_cell_6/split/split_dim
while/lstm_cell_6/splitSplit*while/lstm_cell_6/split/split_dim:output:0"while/lstm_cell_6/BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿn:ÿÿÿÿÿÿÿÿÿn:ÿÿÿÿÿÿÿÿÿn:ÿÿÿÿÿÿÿÿÿn*
	num_split2
while/lstm_cell_6/split
while/lstm_cell_6/SigmoidSigmoid while/lstm_cell_6/split:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿn2
while/lstm_cell_6/Sigmoid
while/lstm_cell_6/Sigmoid_1Sigmoid while/lstm_cell_6/split:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿn2
while/lstm_cell_6/Sigmoid_1
while/lstm_cell_6/mulMulwhile/lstm_cell_6/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿn2
while/lstm_cell_6/mul
while/lstm_cell_6/ReluRelu while/lstm_cell_6/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿn2
while/lstm_cell_6/Relu°
while/lstm_cell_6/mul_1Mulwhile/lstm_cell_6/Sigmoid:y:0$while/lstm_cell_6/Relu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿn2
while/lstm_cell_6/mul_1¥
while/lstm_cell_6/add_1AddV2while/lstm_cell_6/mul:z:0while/lstm_cell_6/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿn2
while/lstm_cell_6/add_1
while/lstm_cell_6/Sigmoid_2Sigmoid while/lstm_cell_6/split:output:3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿn2
while/lstm_cell_6/Sigmoid_2
while/lstm_cell_6/Relu_1Reluwhile/lstm_cell_6/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿn2
while/lstm_cell_6/Relu_1´
while/lstm_cell_6/mul_2Mulwhile/lstm_cell_6/Sigmoid_2:y:0&while/lstm_cell_6/Relu_1:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿn2
while/lstm_cell_6/mul_2ß
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_6/mul_2:z:0*
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
while/add_1ß
while/IdentityIdentitywhile/add_1:z:0)^while/lstm_cell_6/BiasAdd/ReadVariableOp(^while/lstm_cell_6/MatMul/ReadVariableOp*^while/lstm_cell_6/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identityò
while/Identity_1Identitywhile_while_maximum_iterations)^while/lstm_cell_6/BiasAdd/ReadVariableOp(^while/lstm_cell_6/MatMul/ReadVariableOp*^while/lstm_cell_6/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_1á
while/Identity_2Identitywhile/add:z:0)^while/lstm_cell_6/BiasAdd/ReadVariableOp(^while/lstm_cell_6/MatMul/ReadVariableOp*^while/lstm_cell_6/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_2
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0)^while/lstm_cell_6/BiasAdd/ReadVariableOp(^while/lstm_cell_6/MatMul/ReadVariableOp*^while/lstm_cell_6/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_3
while/Identity_4Identitywhile/lstm_cell_6/mul_2:z:0)^while/lstm_cell_6/BiasAdd/ReadVariableOp(^while/lstm_cell_6/MatMul/ReadVariableOp*^while/lstm_cell_6/MatMul_1/ReadVariableOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿn2
while/Identity_4
while/Identity_5Identitywhile/lstm_cell_6/add_1:z:0)^while/lstm_cell_6/BiasAdd/ReadVariableOp(^while/lstm_cell_6/MatMul/ReadVariableOp*^while/lstm_cell_6/MatMul_1/ReadVariableOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿn2
while/Identity_5")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"h
1while_lstm_cell_6_biasadd_readvariableop_resource3while_lstm_cell_6_biasadd_readvariableop_resource_0"j
2while_lstm_cell_6_matmul_1_readvariableop_resource4while_lstm_cell_6_matmul_1_readvariableop_resource_0"f
0while_lstm_cell_6_matmul_readvariableop_resource2while_lstm_cell_6_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*Q
_input_shapes@
>: : : : :ÿÿÿÿÿÿÿÿÿn:ÿÿÿÿÿÿÿÿÿn: : :::2T
(while/lstm_cell_6/BiasAdd/ReadVariableOp(while/lstm_cell_6/BiasAdd/ReadVariableOp2R
'while/lstm_cell_6/MatMul/ReadVariableOp'while/lstm_cell_6/MatMul/ReadVariableOp2V
)while/lstm_cell_6/MatMul_1/ReadVariableOp)while/lstm_cell_6/MatMul_1/ReadVariableOp: 
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
:ÿÿÿÿÿÿÿÿÿn:-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿn:

_output_shapes
: :

_output_shapes
: 
Â6


__inference__traced_save_63639
file_prefix-
)savev2_dense_6_kernel_read_readvariableop+
'savev2_dense_6_bias_read_readvariableop(
$savev2_adam_iter_read_readvariableop	*
&savev2_adam_beta_1_read_readvariableop*
&savev2_adam_beta_2_read_readvariableop)
%savev2_adam_decay_read_readvariableop1
-savev2_adam_learning_rate_read_readvariableop8
4savev2_lstm_6_lstm_cell_6_kernel_read_readvariableopB
>savev2_lstm_6_lstm_cell_6_recurrent_kernel_read_readvariableop6
2savev2_lstm_6_lstm_cell_6_bias_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop4
0savev2_adam_dense_6_kernel_m_read_readvariableop2
.savev2_adam_dense_6_bias_m_read_readvariableop?
;savev2_adam_lstm_6_lstm_cell_6_kernel_m_read_readvariableopI
Esavev2_adam_lstm_6_lstm_cell_6_recurrent_kernel_m_read_readvariableop=
9savev2_adam_lstm_6_lstm_cell_6_bias_m_read_readvariableop4
0savev2_adam_dense_6_kernel_v_read_readvariableop2
.savev2_adam_dense_6_bias_v_read_readvariableop?
;savev2_adam_lstm_6_lstm_cell_6_kernel_v_read_readvariableopI
Esavev2_adam_lstm_6_lstm_cell_6_recurrent_kernel_v_read_readvariableop=
9savev2_adam_lstm_6_lstm_cell_6_bias_v_read_readvariableop
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
SaveV2/shape_and_slices 

SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0)savev2_dense_6_kernel_read_readvariableop'savev2_dense_6_bias_read_readvariableop$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop-savev2_adam_learning_rate_read_readvariableop4savev2_lstm_6_lstm_cell_6_kernel_read_readvariableop>savev2_lstm_6_lstm_cell_6_recurrent_kernel_read_readvariableop2savev2_lstm_6_lstm_cell_6_bias_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop0savev2_adam_dense_6_kernel_m_read_readvariableop.savev2_adam_dense_6_bias_m_read_readvariableop;savev2_adam_lstm_6_lstm_cell_6_kernel_m_read_readvariableopEsavev2_adam_lstm_6_lstm_cell_6_recurrent_kernel_m_read_readvariableop9savev2_adam_lstm_6_lstm_cell_6_bias_m_read_readvariableop0savev2_adam_dense_6_kernel_v_read_readvariableop.savev2_adam_dense_6_bias_v_read_readvariableop;savev2_adam_lstm_6_lstm_cell_6_kernel_v_read_readvariableopEsavev2_adam_lstm_6_lstm_cell_6_recurrent_kernel_v_read_readvariableop9savev2_adam_lstm_6_lstm_cell_6_bias_v_read_readvariableopsavev2_const"/device:CPU:0*
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
: :n:: : : : : :	¸:	n¸:¸: : :n::	¸:	n¸:¸:n::	¸:	n¸:¸: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:$ 

_output_shapes

:n: 
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
:	¸:%	!

_output_shapes
:	n¸:!


_output_shapes	
:¸:

_output_shapes
: :

_output_shapes
: :$ 

_output_shapes

:n: 

_output_shapes
::%!

_output_shapes
:	¸:%!

_output_shapes
:	n¸:!

_output_shapes	
:¸:$ 

_output_shapes

:n: 

_output_shapes
::%!

_output_shapes
:	¸:%!

_output_shapes
:	n¸:!

_output_shapes	
:¸:

_output_shapes
: 
ù	
Ê
lstm_6_while_cond_62653*
&lstm_6_while_lstm_6_while_loop_counter0
,lstm_6_while_lstm_6_while_maximum_iterations
lstm_6_while_placeholder
lstm_6_while_placeholder_1
lstm_6_while_placeholder_2
lstm_6_while_placeholder_3,
(lstm_6_while_less_lstm_6_strided_slice_1A
=lstm_6_while_lstm_6_while_cond_62653___redundant_placeholder0A
=lstm_6_while_lstm_6_while_cond_62653___redundant_placeholder1A
=lstm_6_while_lstm_6_while_cond_62653___redundant_placeholder2A
=lstm_6_while_lstm_6_while_cond_62653___redundant_placeholder3
lstm_6_while_identity

lstm_6/while/LessLesslstm_6_while_placeholder(lstm_6_while_less_lstm_6_strided_slice_1*
T0*
_output_shapes
: 2
lstm_6/while/Lessr
lstm_6/while/IdentityIdentitylstm_6/while/Less:z:0*
T0
*
_output_shapes
: 2
lstm_6/while/Identity"7
lstm_6_while_identitylstm_6/while/Identity:output:0*S
_input_shapesB
@: : : : :ÿÿÿÿÿÿÿÿÿn:ÿÿÿÿÿÿÿÿÿn: ::::: 
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
:ÿÿÿÿÿÿÿÿÿn:-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿn:

_output_shapes
: :

_output_shapes
:
¦
¾
while_cond_62995
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_13
/while_while_cond_62995___redundant_placeholder03
/while_while_cond_62995___redundant_placeholder13
/while_while_cond_62995___redundant_placeholder23
/while_while_cond_62995___redundant_placeholder3
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
@: : : : :ÿÿÿÿÿÿÿÿÿn:ÿÿÿÿÿÿÿÿÿn: ::::: 
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
:ÿÿÿÿÿÿÿÿÿn:-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿn:

_output_shapes
: :

_output_shapes
:
¾Z
ê
A__inference_lstm_6_layer_call_and_return_conditional_losses_62266

inputs.
*lstm_cell_6_matmul_readvariableop_resource0
,lstm_cell_6_matmul_1_readvariableop_resource/
+lstm_cell_6_biasadd_readvariableop_resource
identity¢"lstm_cell_6/BiasAdd/ReadVariableOp¢!lstm_cell_6/MatMul/ReadVariableOp¢#lstm_cell_6/MatMul_1/ReadVariableOp¢whileD
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
value	B :n2
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
value	B :n2
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
:ÿÿÿÿÿÿÿÿÿn2
zeros`
zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B :n2
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
value	B :n2
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
:ÿÿÿÿÿÿÿÿÿn2	
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
:ÿÿÿÿÿÿÿÿÿ2
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
strided_slice_2²
!lstm_cell_6/MatMul/ReadVariableOpReadVariableOp*lstm_cell_6_matmul_readvariableop_resource*
_output_shapes
:	¸*
dtype02#
!lstm_cell_6/MatMul/ReadVariableOpª
lstm_cell_6/MatMulMatMulstrided_slice_2:output:0)lstm_cell_6/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¸2
lstm_cell_6/MatMul¸
#lstm_cell_6/MatMul_1/ReadVariableOpReadVariableOp,lstm_cell_6_matmul_1_readvariableop_resource*
_output_shapes
:	n¸*
dtype02%
#lstm_cell_6/MatMul_1/ReadVariableOp¦
lstm_cell_6/MatMul_1MatMulzeros:output:0+lstm_cell_6/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¸2
lstm_cell_6/MatMul_1
lstm_cell_6/addAddV2lstm_cell_6/MatMul:product:0lstm_cell_6/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¸2
lstm_cell_6/add±
"lstm_cell_6/BiasAdd/ReadVariableOpReadVariableOp+lstm_cell_6_biasadd_readvariableop_resource*
_output_shapes	
:¸*
dtype02$
"lstm_cell_6/BiasAdd/ReadVariableOp©
lstm_cell_6/BiasAddBiasAddlstm_cell_6/add:z:0*lstm_cell_6/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¸2
lstm_cell_6/BiasAddh
lstm_cell_6/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_6/Const|
lstm_cell_6/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_6/split/split_dimï
lstm_cell_6/splitSplit$lstm_cell_6/split/split_dim:output:0lstm_cell_6/BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿn:ÿÿÿÿÿÿÿÿÿn:ÿÿÿÿÿÿÿÿÿn:ÿÿÿÿÿÿÿÿÿn*
	num_split2
lstm_cell_6/split
lstm_cell_6/SigmoidSigmoidlstm_cell_6/split:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿn2
lstm_cell_6/Sigmoid
lstm_cell_6/Sigmoid_1Sigmoidlstm_cell_6/split:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿn2
lstm_cell_6/Sigmoid_1
lstm_cell_6/mulMullstm_cell_6/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿn2
lstm_cell_6/mulz
lstm_cell_6/ReluRelulstm_cell_6/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿn2
lstm_cell_6/Relu
lstm_cell_6/mul_1Mullstm_cell_6/Sigmoid:y:0lstm_cell_6/Relu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿn2
lstm_cell_6/mul_1
lstm_cell_6/add_1AddV2lstm_cell_6/mul:z:0lstm_cell_6/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿn2
lstm_cell_6/add_1
lstm_cell_6/Sigmoid_2Sigmoidlstm_cell_6/split:output:3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿn2
lstm_cell_6/Sigmoid_2y
lstm_cell_6/Relu_1Relulstm_cell_6/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿn2
lstm_cell_6/Relu_1
lstm_cell_6/mul_2Mullstm_cell_6/Sigmoid_2:y:0 lstm_cell_6/Relu_1:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿn2
lstm_cell_6/mul_2
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿn   2
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
while/loop_counteré
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0*lstm_cell_6_matmul_readvariableop_resource,lstm_cell_6_matmul_1_readvariableop_resource+lstm_cell_6_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :ÿÿÿÿÿÿÿÿÿn:ÿÿÿÿÿÿÿÿÿn: : : : : *%
_read_only_resource_inputs
	
*
bodyR
while_body_62181*
condR
while_cond_62180*K
output_shapes:
8: : : : :ÿÿÿÿÿÿÿÿÿn:ÿÿÿÿÿÿÿÿÿn: : : : : *
parallel_iterations 2
whileµ
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿn   22
0TensorArrayV2Stack/TensorListStack/element_shapeè
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿn*
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
:ÿÿÿÿÿÿÿÿÿn*
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
:ÿÿÿÿÿÿÿÿÿn2
transpose_1f
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2	
runtimeã
IdentityIdentitystrided_slice_3:output:0#^lstm_cell_6/BiasAdd/ReadVariableOp"^lstm_cell_6/MatMul/ReadVariableOp$^lstm_cell_6/MatMul_1/ReadVariableOp^while*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿn2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:ÿÿÿÿÿÿÿÿÿ:::2H
"lstm_cell_6/BiasAdd/ReadVariableOp"lstm_cell_6/BiasAdd/ReadVariableOp2F
!lstm_cell_6/MatMul/ReadVariableOp!lstm_cell_6/MatMul/ReadVariableOp2J
#lstm_cell_6/MatMul_1/ReadVariableOp#lstm_cell_6/MatMul_1/ReadVariableOp2
whilewhile:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
à

G__inference_sequential_6_layer_call_and_return_conditional_losses_62323
lstm_6_input
lstm_6_62289
lstm_6_62291
lstm_6_62293
dense_6_62317
dense_6_62319
identity¢dense_6/StatefulPartitionedCall¢lstm_6/StatefulPartitionedCall
lstm_6/StatefulPartitionedCallStatefulPartitionedCalllstm_6_inputlstm_6_62289lstm_6_62291lstm_6_62293*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿn*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *J
fERC
A__inference_lstm_6_layer_call_and_return_conditional_losses_621132 
lstm_6/StatefulPartitionedCall­
dense_6/StatefulPartitionedCallStatefulPartitionedCall'lstm_6/StatefulPartitionedCall:output:0dense_6_62317dense_6_62319*
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
GPU 2J 8 *K
fFRD
B__inference_dense_6_layer_call_and_return_conditional_losses_623062!
dense_6/StatefulPartitionedCall¿
IdentityIdentity(dense_6/StatefulPartitionedCall:output:0 ^dense_6/StatefulPartitionedCall^lstm_6/StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:ÿÿÿÿÿÿÿÿÿ:::::2B
dense_6/StatefulPartitionedCalldense_6/StatefulPartitionedCall2@
lstm_6/StatefulPartitionedCalllstm_6/StatefulPartitionedCall:Y U
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
&
_user_specified_namelstm_6_input
ùo
»
G__inference_sequential_6_layer_call_and_return_conditional_losses_62586

inputs5
1lstm_6_lstm_cell_6_matmul_readvariableop_resource7
3lstm_6_lstm_cell_6_matmul_1_readvariableop_resource6
2lstm_6_lstm_cell_6_biasadd_readvariableop_resource*
&dense_6_matmul_readvariableop_resource+
'dense_6_biasadd_readvariableop_resource
identity¢dense_6/BiasAdd/ReadVariableOp¢dense_6/MatMul/ReadVariableOp¢)lstm_6/lstm_cell_6/BiasAdd/ReadVariableOp¢(lstm_6/lstm_cell_6/MatMul/ReadVariableOp¢*lstm_6/lstm_cell_6/MatMul_1/ReadVariableOp¢lstm_6/whileR
lstm_6/ShapeShapeinputs*
T0*
_output_shapes
:2
lstm_6/Shape
lstm_6/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
lstm_6/strided_slice/stack
lstm_6/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
lstm_6/strided_slice/stack_1
lstm_6/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
lstm_6/strided_slice/stack_2
lstm_6/strided_sliceStridedSlicelstm_6/Shape:output:0#lstm_6/strided_slice/stack:output:0%lstm_6/strided_slice/stack_1:output:0%lstm_6/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
lstm_6/strided_slicej
lstm_6/zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :n2
lstm_6/zeros/mul/y
lstm_6/zeros/mulMullstm_6/strided_slice:output:0lstm_6/zeros/mul/y:output:0*
T0*
_output_shapes
: 2
lstm_6/zeros/mulm
lstm_6/zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :è2
lstm_6/zeros/Less/y
lstm_6/zeros/LessLesslstm_6/zeros/mul:z:0lstm_6/zeros/Less/y:output:0*
T0*
_output_shapes
: 2
lstm_6/zeros/Lessp
lstm_6/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :n2
lstm_6/zeros/packed/1
lstm_6/zeros/packedPacklstm_6/strided_slice:output:0lstm_6/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
lstm_6/zeros/packedm
lstm_6/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
lstm_6/zeros/Const
lstm_6/zerosFilllstm_6/zeros/packed:output:0lstm_6/zeros/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿn2
lstm_6/zerosn
lstm_6/zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B :n2
lstm_6/zeros_1/mul/y
lstm_6/zeros_1/mulMullstm_6/strided_slice:output:0lstm_6/zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
lstm_6/zeros_1/mulq
lstm_6/zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :è2
lstm_6/zeros_1/Less/y
lstm_6/zeros_1/LessLesslstm_6/zeros_1/mul:z:0lstm_6/zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
lstm_6/zeros_1/Lesst
lstm_6/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :n2
lstm_6/zeros_1/packed/1¥
lstm_6/zeros_1/packedPacklstm_6/strided_slice:output:0 lstm_6/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
lstm_6/zeros_1/packedq
lstm_6/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
lstm_6/zeros_1/Const
lstm_6/zeros_1Filllstm_6/zeros_1/packed:output:0lstm_6/zeros_1/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿn2
lstm_6/zeros_1
lstm_6/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
lstm_6/transpose/perm
lstm_6/transpose	Transposeinputslstm_6/transpose/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_6/transposed
lstm_6/Shape_1Shapelstm_6/transpose:y:0*
T0*
_output_shapes
:2
lstm_6/Shape_1
lstm_6/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
lstm_6/strided_slice_1/stack
lstm_6/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2 
lstm_6/strided_slice_1/stack_1
lstm_6/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2 
lstm_6/strided_slice_1/stack_2
lstm_6/strided_slice_1StridedSlicelstm_6/Shape_1:output:0%lstm_6/strided_slice_1/stack:output:0'lstm_6/strided_slice_1/stack_1:output:0'lstm_6/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
lstm_6/strided_slice_1
"lstm_6/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2$
"lstm_6/TensorArrayV2/element_shapeÎ
lstm_6/TensorArrayV2TensorListReserve+lstm_6/TensorArrayV2/element_shape:output:0lstm_6/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
lstm_6/TensorArrayV2Í
<lstm_6/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   2>
<lstm_6/TensorArrayUnstack/TensorListFromTensor/element_shape
.lstm_6/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorlstm_6/transpose:y:0Elstm_6/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type020
.lstm_6/TensorArrayUnstack/TensorListFromTensor
lstm_6/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
lstm_6/strided_slice_2/stack
lstm_6/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2 
lstm_6/strided_slice_2/stack_1
lstm_6/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2 
lstm_6/strided_slice_2/stack_2¦
lstm_6/strided_slice_2StridedSlicelstm_6/transpose:y:0%lstm_6/strided_slice_2/stack:output:0'lstm_6/strided_slice_2/stack_1:output:0'lstm_6/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
shrink_axis_mask2
lstm_6/strided_slice_2Ç
(lstm_6/lstm_cell_6/MatMul/ReadVariableOpReadVariableOp1lstm_6_lstm_cell_6_matmul_readvariableop_resource*
_output_shapes
:	¸*
dtype02*
(lstm_6/lstm_cell_6/MatMul/ReadVariableOpÆ
lstm_6/lstm_cell_6/MatMulMatMullstm_6/strided_slice_2:output:00lstm_6/lstm_cell_6/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¸2
lstm_6/lstm_cell_6/MatMulÍ
*lstm_6/lstm_cell_6/MatMul_1/ReadVariableOpReadVariableOp3lstm_6_lstm_cell_6_matmul_1_readvariableop_resource*
_output_shapes
:	n¸*
dtype02,
*lstm_6/lstm_cell_6/MatMul_1/ReadVariableOpÂ
lstm_6/lstm_cell_6/MatMul_1MatMullstm_6/zeros:output:02lstm_6/lstm_cell_6/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¸2
lstm_6/lstm_cell_6/MatMul_1¸
lstm_6/lstm_cell_6/addAddV2#lstm_6/lstm_cell_6/MatMul:product:0%lstm_6/lstm_cell_6/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¸2
lstm_6/lstm_cell_6/addÆ
)lstm_6/lstm_cell_6/BiasAdd/ReadVariableOpReadVariableOp2lstm_6_lstm_cell_6_biasadd_readvariableop_resource*
_output_shapes	
:¸*
dtype02+
)lstm_6/lstm_cell_6/BiasAdd/ReadVariableOpÅ
lstm_6/lstm_cell_6/BiasAddBiasAddlstm_6/lstm_cell_6/add:z:01lstm_6/lstm_cell_6/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¸2
lstm_6/lstm_cell_6/BiasAddv
lstm_6/lstm_cell_6/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_6/lstm_cell_6/Const
"lstm_6/lstm_cell_6/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2$
"lstm_6/lstm_cell_6/split/split_dim
lstm_6/lstm_cell_6/splitSplit+lstm_6/lstm_cell_6/split/split_dim:output:0#lstm_6/lstm_cell_6/BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿn:ÿÿÿÿÿÿÿÿÿn:ÿÿÿÿÿÿÿÿÿn:ÿÿÿÿÿÿÿÿÿn*
	num_split2
lstm_6/lstm_cell_6/split
lstm_6/lstm_cell_6/SigmoidSigmoid!lstm_6/lstm_cell_6/split:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿn2
lstm_6/lstm_cell_6/Sigmoid
lstm_6/lstm_cell_6/Sigmoid_1Sigmoid!lstm_6/lstm_cell_6/split:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿn2
lstm_6/lstm_cell_6/Sigmoid_1¤
lstm_6/lstm_cell_6/mulMul lstm_6/lstm_cell_6/Sigmoid_1:y:0lstm_6/zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿn2
lstm_6/lstm_cell_6/mul
lstm_6/lstm_cell_6/ReluRelu!lstm_6/lstm_cell_6/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿn2
lstm_6/lstm_cell_6/Relu´
lstm_6/lstm_cell_6/mul_1Mullstm_6/lstm_cell_6/Sigmoid:y:0%lstm_6/lstm_cell_6/Relu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿn2
lstm_6/lstm_cell_6/mul_1©
lstm_6/lstm_cell_6/add_1AddV2lstm_6/lstm_cell_6/mul:z:0lstm_6/lstm_cell_6/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿn2
lstm_6/lstm_cell_6/add_1
lstm_6/lstm_cell_6/Sigmoid_2Sigmoid!lstm_6/lstm_cell_6/split:output:3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿn2
lstm_6/lstm_cell_6/Sigmoid_2
lstm_6/lstm_cell_6/Relu_1Relulstm_6/lstm_cell_6/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿn2
lstm_6/lstm_cell_6/Relu_1¸
lstm_6/lstm_cell_6/mul_2Mul lstm_6/lstm_cell_6/Sigmoid_2:y:0'lstm_6/lstm_cell_6/Relu_1:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿn2
lstm_6/lstm_cell_6/mul_2
$lstm_6/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿn   2&
$lstm_6/TensorArrayV2_1/element_shapeÔ
lstm_6/TensorArrayV2_1TensorListReserve-lstm_6/TensorArrayV2_1/element_shape:output:0lstm_6/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
lstm_6/TensorArrayV2_1\
lstm_6/timeConst*
_output_shapes
: *
dtype0*
value	B : 2
lstm_6/time
lstm_6/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2!
lstm_6/while/maximum_iterationsx
lstm_6/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
lstm_6/while/loop_counterÒ
lstm_6/whileWhile"lstm_6/while/loop_counter:output:0(lstm_6/while/maximum_iterations:output:0lstm_6/time:output:0lstm_6/TensorArrayV2_1:handle:0lstm_6/zeros:output:0lstm_6/zeros_1:output:0lstm_6/strided_slice_1:output:0>lstm_6/TensorArrayUnstack/TensorListFromTensor:output_handle:01lstm_6_lstm_cell_6_matmul_readvariableop_resource3lstm_6_lstm_cell_6_matmul_1_readvariableop_resource2lstm_6_lstm_cell_6_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :ÿÿÿÿÿÿÿÿÿn:ÿÿÿÿÿÿÿÿÿn: : : : : *%
_read_only_resource_inputs
	
*#
bodyR
lstm_6_while_body_62495*#
condR
lstm_6_while_cond_62494*K
output_shapes:
8: : : : :ÿÿÿÿÿÿÿÿÿn:ÿÿÿÿÿÿÿÿÿn: : : : : *
parallel_iterations 2
lstm_6/whileÃ
7lstm_6/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿn   29
7lstm_6/TensorArrayV2Stack/TensorListStack/element_shape
)lstm_6/TensorArrayV2Stack/TensorListStackTensorListStacklstm_6/while:output:3@lstm_6/TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿn*
element_dtype02+
)lstm_6/TensorArrayV2Stack/TensorListStack
lstm_6/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿ2
lstm_6/strided_slice_3/stack
lstm_6/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2 
lstm_6/strided_slice_3/stack_1
lstm_6/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2 
lstm_6/strided_slice_3/stack_2Ä
lstm_6/strided_slice_3StridedSlice2lstm_6/TensorArrayV2Stack/TensorListStack:tensor:0%lstm_6/strided_slice_3/stack:output:0'lstm_6/strided_slice_3/stack_1:output:0'lstm_6/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿn*
shrink_axis_mask2
lstm_6/strided_slice_3
lstm_6/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
lstm_6/transpose_1/permÁ
lstm_6/transpose_1	Transpose2lstm_6/TensorArrayV2Stack/TensorListStack:tensor:0 lstm_6/transpose_1/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿn2
lstm_6/transpose_1t
lstm_6/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2
lstm_6/runtime¥
dense_6/MatMul/ReadVariableOpReadVariableOp&dense_6_matmul_readvariableop_resource*
_output_shapes

:n*
dtype02
dense_6/MatMul/ReadVariableOp¤
dense_6/MatMulMatMullstm_6/strided_slice_3:output:0%dense_6/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_6/MatMul¤
dense_6/BiasAdd/ReadVariableOpReadVariableOp'dense_6_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02 
dense_6/BiasAdd/ReadVariableOp¡
dense_6/BiasAddBiasAdddense_6/MatMul:product:0&dense_6/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_6/BiasAddÀ
IdentityIdentitydense_6/BiasAdd:output:0^dense_6/BiasAdd/ReadVariableOp^dense_6/MatMul/ReadVariableOp*^lstm_6/lstm_cell_6/BiasAdd/ReadVariableOp)^lstm_6/lstm_cell_6/MatMul/ReadVariableOp+^lstm_6/lstm_cell_6/MatMul_1/ReadVariableOp^lstm_6/while*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:ÿÿÿÿÿÿÿÿÿ:::::2@
dense_6/BiasAdd/ReadVariableOpdense_6/BiasAdd/ReadVariableOp2>
dense_6/MatMul/ReadVariableOpdense_6/MatMul/ReadVariableOp2V
)lstm_6/lstm_cell_6/BiasAdd/ReadVariableOp)lstm_6/lstm_cell_6/BiasAdd/ReadVariableOp2T
(lstm_6/lstm_cell_6/MatMul/ReadVariableOp(lstm_6/lstm_cell_6/MatMul/ReadVariableOp2X
*lstm_6/lstm_cell_6/MatMul_1/ReadVariableOp*lstm_6/lstm_cell_6/MatMul_1/ReadVariableOp2
lstm_6/whilelstm_6/while:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ÿ

&__inference_lstm_6_layer_call_fn_63431

inputs
unknown
	unknown_0
	unknown_1
identity¢StatefulPartitionedCallþ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿn*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *J
fERC
A__inference_lstm_6_layer_call_and_return_conditional_losses_622662
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿn2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:ÿÿÿÿÿÿÿÿÿ:::22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
B
ô
while_body_62996
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_06
2while_lstm_cell_6_matmul_readvariableop_resource_08
4while_lstm_cell_6_matmul_1_readvariableop_resource_07
3while_lstm_cell_6_biasadd_readvariableop_resource_0
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor4
0while_lstm_cell_6_matmul_readvariableop_resource6
2while_lstm_cell_6_matmul_1_readvariableop_resource5
1while_lstm_cell_6_biasadd_readvariableop_resource¢(while/lstm_cell_6/BiasAdd/ReadVariableOp¢'while/lstm_cell_6/MatMul/ReadVariableOp¢)while/lstm_cell_6/MatMul_1/ReadVariableOpÃ
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
)while/TensorArrayV2Read/TensorListGetItemÆ
'while/lstm_cell_6/MatMul/ReadVariableOpReadVariableOp2while_lstm_cell_6_matmul_readvariableop_resource_0*
_output_shapes
:	¸*
dtype02)
'while/lstm_cell_6/MatMul/ReadVariableOpÔ
while/lstm_cell_6/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0/while/lstm_cell_6/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¸2
while/lstm_cell_6/MatMulÌ
)while/lstm_cell_6/MatMul_1/ReadVariableOpReadVariableOp4while_lstm_cell_6_matmul_1_readvariableop_resource_0*
_output_shapes
:	n¸*
dtype02+
)while/lstm_cell_6/MatMul_1/ReadVariableOp½
while/lstm_cell_6/MatMul_1MatMulwhile_placeholder_21while/lstm_cell_6/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¸2
while/lstm_cell_6/MatMul_1´
while/lstm_cell_6/addAddV2"while/lstm_cell_6/MatMul:product:0$while/lstm_cell_6/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¸2
while/lstm_cell_6/addÅ
(while/lstm_cell_6/BiasAdd/ReadVariableOpReadVariableOp3while_lstm_cell_6_biasadd_readvariableop_resource_0*
_output_shapes	
:¸*
dtype02*
(while/lstm_cell_6/BiasAdd/ReadVariableOpÁ
while/lstm_cell_6/BiasAddBiasAddwhile/lstm_cell_6/add:z:00while/lstm_cell_6/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¸2
while/lstm_cell_6/BiasAddt
while/lstm_cell_6/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
while/lstm_cell_6/Const
!while/lstm_cell_6/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2#
!while/lstm_cell_6/split/split_dim
while/lstm_cell_6/splitSplit*while/lstm_cell_6/split/split_dim:output:0"while/lstm_cell_6/BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿn:ÿÿÿÿÿÿÿÿÿn:ÿÿÿÿÿÿÿÿÿn:ÿÿÿÿÿÿÿÿÿn*
	num_split2
while/lstm_cell_6/split
while/lstm_cell_6/SigmoidSigmoid while/lstm_cell_6/split:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿn2
while/lstm_cell_6/Sigmoid
while/lstm_cell_6/Sigmoid_1Sigmoid while/lstm_cell_6/split:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿn2
while/lstm_cell_6/Sigmoid_1
while/lstm_cell_6/mulMulwhile/lstm_cell_6/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿn2
while/lstm_cell_6/mul
while/lstm_cell_6/ReluRelu while/lstm_cell_6/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿn2
while/lstm_cell_6/Relu°
while/lstm_cell_6/mul_1Mulwhile/lstm_cell_6/Sigmoid:y:0$while/lstm_cell_6/Relu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿn2
while/lstm_cell_6/mul_1¥
while/lstm_cell_6/add_1AddV2while/lstm_cell_6/mul:z:0while/lstm_cell_6/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿn2
while/lstm_cell_6/add_1
while/lstm_cell_6/Sigmoid_2Sigmoid while/lstm_cell_6/split:output:3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿn2
while/lstm_cell_6/Sigmoid_2
while/lstm_cell_6/Relu_1Reluwhile/lstm_cell_6/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿn2
while/lstm_cell_6/Relu_1´
while/lstm_cell_6/mul_2Mulwhile/lstm_cell_6/Sigmoid_2:y:0&while/lstm_cell_6/Relu_1:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿn2
while/lstm_cell_6/mul_2ß
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_6/mul_2:z:0*
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
while/add_1ß
while/IdentityIdentitywhile/add_1:z:0)^while/lstm_cell_6/BiasAdd/ReadVariableOp(^while/lstm_cell_6/MatMul/ReadVariableOp*^while/lstm_cell_6/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identityò
while/Identity_1Identitywhile_while_maximum_iterations)^while/lstm_cell_6/BiasAdd/ReadVariableOp(^while/lstm_cell_6/MatMul/ReadVariableOp*^while/lstm_cell_6/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_1á
while/Identity_2Identitywhile/add:z:0)^while/lstm_cell_6/BiasAdd/ReadVariableOp(^while/lstm_cell_6/MatMul/ReadVariableOp*^while/lstm_cell_6/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_2
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0)^while/lstm_cell_6/BiasAdd/ReadVariableOp(^while/lstm_cell_6/MatMul/ReadVariableOp*^while/lstm_cell_6/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_3
while/Identity_4Identitywhile/lstm_cell_6/mul_2:z:0)^while/lstm_cell_6/BiasAdd/ReadVariableOp(^while/lstm_cell_6/MatMul/ReadVariableOp*^while/lstm_cell_6/MatMul_1/ReadVariableOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿn2
while/Identity_4
while/Identity_5Identitywhile/lstm_cell_6/add_1:z:0)^while/lstm_cell_6/BiasAdd/ReadVariableOp(^while/lstm_cell_6/MatMul/ReadVariableOp*^while/lstm_cell_6/MatMul_1/ReadVariableOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿn2
while/Identity_5")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"h
1while_lstm_cell_6_biasadd_readvariableop_resource3while_lstm_cell_6_biasadd_readvariableop_resource_0"j
2while_lstm_cell_6_matmul_1_readvariableop_resource4while_lstm_cell_6_matmul_1_readvariableop_resource_0"f
0while_lstm_cell_6_matmul_readvariableop_resource2while_lstm_cell_6_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*Q
_input_shapes@
>: : : : :ÿÿÿÿÿÿÿÿÿn:ÿÿÿÿÿÿÿÿÿn: : :::2T
(while/lstm_cell_6/BiasAdd/ReadVariableOp(while/lstm_cell_6/BiasAdd/ReadVariableOp2R
'while/lstm_cell_6/MatMul/ReadVariableOp'while/lstm_cell_6/MatMul/ReadVariableOp2V
)while/lstm_cell_6/MatMul_1/ReadVariableOp)while/lstm_cell_6/MatMul_1/ReadVariableOp: 
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
:ÿÿÿÿÿÿÿÿÿn:-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿn:

_output_shapes
: :

_output_shapes
: 
¦
¾
while_cond_62027
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_13
/while_while_cond_62027___redundant_placeholder03
/while_while_cond_62027___redundant_placeholder13
/while_while_cond_62027___redundant_placeholder23
/while_while_cond_62027___redundant_placeholder3
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
@: : : : :ÿÿÿÿÿÿÿÿÿn:ÿÿÿÿÿÿÿÿÿn: ::::: 
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
:ÿÿÿÿÿÿÿÿÿn:-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿn:

_output_shapes
: :

_output_shapes
:
©
Ù
F__inference_lstm_cell_6_layer_call_and_return_conditional_losses_61420

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
:	¸*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¸2
MatMul
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource*
_output_shapes
:	n¸*
dtype02
MatMul_1/ReadVariableOpz
MatMul_1MatMulstatesMatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¸2

MatMul_1l
addAddV2MatMul:product:0MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¸2
add
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:¸*
dtype02
BiasAdd/ReadVariableOpy
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¸2	
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
L:ÿÿÿÿÿÿÿÿÿn:ÿÿÿÿÿÿÿÿÿn:ÿÿÿÿÿÿÿÿÿn:ÿÿÿÿÿÿÿÿÿn*
	num_split2
split_
SigmoidSigmoidsplit:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿn2	
Sigmoidc
	Sigmoid_1Sigmoidsplit:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿn2
	Sigmoid_1\
mulMulSigmoid_1:y:0states_1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿn2
mulV
ReluRelusplit:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿn2
Reluh
mul_1MulSigmoid:y:0Relu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿn2
mul_1]
add_1AddV2mul:z:0	mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿn2
add_1c
	Sigmoid_2Sigmoidsplit:output:3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿn2
	Sigmoid_2U
Relu_1Relu	add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿn2
Relu_1l
mul_2MulSigmoid_2:y:0Relu_1:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿn2
mul_2¨
IdentityIdentity	mul_2:z:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿn2

Identity¬

Identity_1Identity	mul_2:z:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿn2

Identity_1¬

Identity_2Identity	add_1:z:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿn2

Identity_2"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*X
_input_shapesG
E:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿn:ÿÿÿÿÿÿÿÿÿn:::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs:OK
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿn
 
_user_specified_namestates:OK
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿn
 
_user_specified_namestates


&__inference_lstm_6_layer_call_fn_63092
inputs_0
unknown
	unknown_0
	unknown_1
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputs_0unknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿn*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *J
fERC
A__inference_lstm_6_layer_call_and_return_conditional_losses_618162
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿn2

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
¦
¾
while_cond_62842
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_13
/while_while_cond_62842___redundant_placeholder03
/while_while_cond_62842___redundant_placeholder13
/while_while_cond_62842___redundant_placeholder23
/while_while_cond_62842___redundant_placeholder3
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
@: : : : :ÿÿÿÿÿÿÿÿÿn:ÿÿÿÿÿÿÿÿÿn: ::::: 
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
:ÿÿÿÿÿÿÿÿÿn:-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿn:

_output_shapes
: :

_output_shapes
:
¦
¾
while_cond_63170
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_13
/while_while_cond_63170___redundant_placeholder03
/while_while_cond_63170___redundant_placeholder13
/while_while_cond_63170___redundant_placeholder23
/while_while_cond_63170___redundant_placeholder3
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
@: : : : :ÿÿÿÿÿÿÿÿÿn:ÿÿÿÿÿÿÿÿÿn: ::::: 
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
:ÿÿÿÿÿÿÿÿÿn:-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿn:

_output_shapes
: :

_output_shapes
:
Î

G__inference_sequential_6_layer_call_and_return_conditional_losses_62389

inputs
lstm_6_62376
lstm_6_62378
lstm_6_62380
dense_6_62383
dense_6_62385
identity¢dense_6/StatefulPartitionedCall¢lstm_6/StatefulPartitionedCall
lstm_6/StatefulPartitionedCallStatefulPartitionedCallinputslstm_6_62376lstm_6_62378lstm_6_62380*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿn*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *J
fERC
A__inference_lstm_6_layer_call_and_return_conditional_losses_622662 
lstm_6/StatefulPartitionedCall­
dense_6/StatefulPartitionedCallStatefulPartitionedCall'lstm_6/StatefulPartitionedCall:output:0dense_6_62383dense_6_62385*
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
GPU 2J 8 *K
fFRD
B__inference_dense_6_layer_call_and_return_conditional_losses_623062!
dense_6/StatefulPartitionedCall¿
IdentityIdentity(dense_6/StatefulPartitionedCall:output:0 ^dense_6/StatefulPartitionedCall^lstm_6/StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:ÿÿÿÿÿÿÿÿÿ:::::2B
dense_6/StatefulPartitionedCalldense_6/StatefulPartitionedCall2@
lstm_6/StatefulPartitionedCalllstm_6/StatefulPartitionedCall:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs"±L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*¸
serving_default¤
I
lstm_6_input9
serving_default_lstm_6_input:0ÿÿÿÿÿÿÿÿÿ;
dense_60
StatefulPartitionedCall:0ÿÿÿÿÿÿÿÿÿtensorflow/serving/predict:÷
"
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
F__call__
*G&call_and_return_all_conditional_losses"ü
_tf_keras_sequentialÝ{"class_name": "Sequential", "name": "sequential_6", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "config": {"name": "sequential_6", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 25, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "lstm_6_input"}}, {"class_name": "LSTM", "config": {"name": "lstm_6", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 25, 1]}, "dtype": "float32", "return_sequences": false, "return_state": false, "go_backwards": false, "stateful": false, "unroll": false, "time_major": false, "units": 110, "activation": "relu", "recurrent_activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "recurrent_initializer": {"class_name": "Orthogonal", "config": {"gain": 1.0, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "unit_forget_bias": true, "kernel_regularizer": null, "recurrent_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "recurrent_constraint": null, "bias_constraint": null, "dropout": 0.0, "recurrent_dropout": 0.0, "implementation": 2}}, {"class_name": "Dense", "config": {"name": "dense_6", "trainable": true, "dtype": "float32", "units": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}]}, "input_spec": [{"class_name": "InputSpec", "config": {"dtype": null, "shape": {"class_name": "__tuple__", "items": [null, null, 1]}, "ndim": 3, "max_ndim": null, "min_ndim": null, "axes": {}}}], "build_input_shape": {"class_name": "TensorShape", "items": [null, 25, 1]}, "is_graph_network": true, "keras_version": "2.4.0", "backend": "tensorflow", "model_config": {"class_name": "Sequential", "config": {"name": "sequential_6", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 25, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "lstm_6_input"}}, {"class_name": "LSTM", "config": {"name": "lstm_6", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 25, 1]}, "dtype": "float32", "return_sequences": false, "return_state": false, "go_backwards": false, "stateful": false, "unroll": false, "time_major": false, "units": 110, "activation": "relu", "recurrent_activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "recurrent_initializer": {"class_name": "Orthogonal", "config": {"gain": 1.0, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "unit_forget_bias": true, "kernel_regularizer": null, "recurrent_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "recurrent_constraint": null, "bias_constraint": null, "dropout": 0.0, "recurrent_dropout": 0.0, "implementation": 2}}, {"class_name": "Dense", "config": {"name": "dense_6", "trainable": true, "dtype": "float32", "units": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}]}}, "training_config": {"loss": "mean_squared_error", "metrics": null, "weighted_metrics": null, "loss_weights": null, "optimizer_config": {"class_name": "Adam", "config": {"name": "Adam", "learning_rate": 0.0010000000474974513, "decay": 0.0, "beta_1": 0.8999999761581421, "beta_2": 0.9990000128746033, "epsilon": 1e-07, "amsgrad": false}}}}
µ
	cell


state_spec
	variables
regularization_losses
trainable_variables
	keras_api
H__call__
*I&call_and_return_all_conditional_losses"
_tf_keras_rnn_layerî
{"class_name": "LSTM", "name": "lstm_6", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": {"class_name": "__tuple__", "items": [null, 25, 1]}, "stateful": false, "must_restore_from_config": false, "config": {"name": "lstm_6", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 25, 1]}, "dtype": "float32", "return_sequences": false, "return_state": false, "go_backwards": false, "stateful": false, "unroll": false, "time_major": false, "units": 110, "activation": "relu", "recurrent_activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "recurrent_initializer": {"class_name": "Orthogonal", "config": {"gain": 1.0, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "unit_forget_bias": true, "kernel_regularizer": null, "recurrent_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "recurrent_constraint": null, "bias_constraint": null, "dropout": 0.0, "recurrent_dropout": 0.0, "implementation": 2}, "input_spec": [{"class_name": "InputSpec", "config": {"dtype": null, "shape": {"class_name": "__tuple__", "items": [null, null, 1]}, "ndim": 3, "max_ndim": null, "min_ndim": null, "axes": {}}}], "build_input_shape": {"class_name": "TensorShape", "items": [null, 25, 1]}}
ó

kernel
bias
regularization_losses
	variables
trainable_variables
	keras_api
J__call__
*K&call_and_return_all_conditional_losses"Î
_tf_keras_layer´{"class_name": "Dense", "name": "dense_6", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_6", "trainable": true, "dtype": "float32", "units": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 110}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 110]}}
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
layer_metrics
	variables
regularization_losses
non_trainable_variables
metrics

 layers
trainable_variables
!layer_regularization_losses
F__call__
E_default_save_signature
*G&call_and_return_all_conditional_losses
&G"call_and_return_conditional_losses"
_generic_user_object
,
Lserving_default"
signature_map
©

kernel
recurrent_kernel
bias
"regularization_losses
#	variables
$trainable_variables
%	keras_api
M__call__
*N&call_and_return_all_conditional_losses"î
_tf_keras_layerÔ{"class_name": "LSTMCell", "name": "lstm_cell_6", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "lstm_cell_6", "trainable": true, "dtype": "float32", "units": 110, "activation": "relu", "recurrent_activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "recurrent_initializer": {"class_name": "Orthogonal", "config": {"gain": 1.0, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "unit_forget_bias": true, "kernel_regularizer": null, "recurrent_regularizer": null, "bias_regularizer": null, "kernel_constraint": null, "recurrent_constraint": null, "bias_constraint": null, "dropout": 0.0, "recurrent_dropout": 0.0, "implementation": 2}}
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
&layer_metrics

'states
	variables
regularization_losses
(non_trainable_variables
)metrics

*layers
trainable_variables
+layer_regularization_losses
H__call__
*I&call_and_return_all_conditional_losses
&I"call_and_return_conditional_losses"
_generic_user_object
 :n2dense_6/kernel
:2dense_6/bias
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
­
,layer_metrics
regularization_losses
	variables
trainable_variables
-non_trainable_variables

.layers
/metrics
0layer_regularization_losses
J__call__
*K&call_and_return_all_conditional_losses
&K"call_and_return_conditional_losses"
_generic_user_object
:	 (2	Adam/iter
: (2Adam/beta_1
: (2Adam/beta_2
: (2
Adam/decay
: (2Adam/learning_rate
,:*	¸2lstm_6/lstm_cell_6/kernel
6:4	n¸2#lstm_6/lstm_cell_6/recurrent_kernel
&:$¸2lstm_6/lstm_cell_6/bias
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
'
10"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
5
0
1
2"
trackable_list_wrapper
5
0
1
2"
trackable_list_wrapper
­
2layer_metrics
"regularization_losses
#	variables
$trainable_variables
3non_trainable_variables

4layers
5metrics
6layer_regularization_losses
M__call__
*N&call_and_return_all_conditional_losses
&N"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
'
	0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
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
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
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
%:#n2Adam/dense_6/kernel/m
:2Adam/dense_6/bias/m
1:/	¸2 Adam/lstm_6/lstm_cell_6/kernel/m
;:9	n¸2*Adam/lstm_6/lstm_cell_6/recurrent_kernel/m
+:)¸2Adam/lstm_6/lstm_cell_6/bias/m
%:#n2Adam/dense_6/kernel/v
:2Adam/dense_6/bias/v
1:/	¸2 Adam/lstm_6/lstm_cell_6/kernel/v
;:9	n¸2*Adam/lstm_6/lstm_cell_6/recurrent_kernel/v
+:)¸2Adam/lstm_6/lstm_cell_6/bias/v
ç2ä
 __inference__wrapped_model_61347¿
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
annotationsª */¢,
*'
lstm_6_inputÿÿÿÿÿÿÿÿÿ
þ2û
,__inference_sequential_6_layer_call_fn_62402
,__inference_sequential_6_layer_call_fn_62775
,__inference_sequential_6_layer_call_fn_62371
,__inference_sequential_6_layer_call_fn_62760À
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
ê2ç
G__inference_sequential_6_layer_call_and_return_conditional_losses_62323
G__inference_sequential_6_layer_call_and_return_conditional_losses_62745
G__inference_sequential_6_layer_call_and_return_conditional_losses_62339
G__inference_sequential_6_layer_call_and_return_conditional_losses_62586À
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
û2ø
&__inference_lstm_6_layer_call_fn_63103
&__inference_lstm_6_layer_call_fn_63420
&__inference_lstm_6_layer_call_fn_63431
&__inference_lstm_6_layer_call_fn_63092Õ
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
ç2ä
A__inference_lstm_6_layer_call_and_return_conditional_losses_62928
A__inference_lstm_6_layer_call_and_return_conditional_losses_63081
A__inference_lstm_6_layer_call_and_return_conditional_losses_63256
A__inference_lstm_6_layer_call_and_return_conditional_losses_63409Õ
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
Ñ2Î
'__inference_dense_6_layer_call_fn_63450¢
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
ì2é
B__inference_dense_6_layer_call_and_return_conditional_losses_63441¢
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
ÏBÌ
#__inference_signature_wrapper_62427lstm_6_input"
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
2
+__inference_lstm_cell_6_layer_call_fn_63533
+__inference_lstm_cell_6_layer_call_fn_63550¾
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
Ô2Ñ
F__inference_lstm_cell_6_layer_call_and_return_conditional_losses_63516
F__inference_lstm_cell_6_layer_call_and_return_conditional_losses_63483¾
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
 
 __inference__wrapped_model_61347u9¢6
/¢,
*'
lstm_6_inputÿÿÿÿÿÿÿÿÿ
ª "1ª.
,
dense_6!
dense_6ÿÿÿÿÿÿÿÿÿ¢
B__inference_dense_6_layer_call_and_return_conditional_losses_63441\/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿn
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 z
'__inference_dense_6_layer_call_fn_63450O/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿn
ª "ÿÿÿÿÿÿÿÿÿÂ
A__inference_lstm_6_layer_call_and_return_conditional_losses_62928}O¢L
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
0ÿÿÿÿÿÿÿÿÿn
 Â
A__inference_lstm_6_layer_call_and_return_conditional_losses_63081}O¢L
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
0ÿÿÿÿÿÿÿÿÿn
 ²
A__inference_lstm_6_layer_call_and_return_conditional_losses_63256m?¢<
5¢2
$!
inputsÿÿÿÿÿÿÿÿÿ

 
p

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿn
 ²
A__inference_lstm_6_layer_call_and_return_conditional_losses_63409m?¢<
5¢2
$!
inputsÿÿÿÿÿÿÿÿÿ

 
p 

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿn
 
&__inference_lstm_6_layer_call_fn_63092pO¢L
E¢B
41
/,
inputs/0ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ

 
p

 
ª "ÿÿÿÿÿÿÿÿÿn
&__inference_lstm_6_layer_call_fn_63103pO¢L
E¢B
41
/,
inputs/0ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ

 
p 

 
ª "ÿÿÿÿÿÿÿÿÿn
&__inference_lstm_6_layer_call_fn_63420`?¢<
5¢2
$!
inputsÿÿÿÿÿÿÿÿÿ

 
p

 
ª "ÿÿÿÿÿÿÿÿÿn
&__inference_lstm_6_layer_call_fn_63431`?¢<
5¢2
$!
inputsÿÿÿÿÿÿÿÿÿ

 
p 

 
ª "ÿÿÿÿÿÿÿÿÿnÈ
F__inference_lstm_cell_6_layer_call_and_return_conditional_losses_63483ý¢}
v¢s
 
inputsÿÿÿÿÿÿÿÿÿ
K¢H
"
states/0ÿÿÿÿÿÿÿÿÿn
"
states/1ÿÿÿÿÿÿÿÿÿn
p
ª "s¢p
i¢f

0/0ÿÿÿÿÿÿÿÿÿn
EB

0/1/0ÿÿÿÿÿÿÿÿÿn

0/1/1ÿÿÿÿÿÿÿÿÿn
 È
F__inference_lstm_cell_6_layer_call_and_return_conditional_losses_63516ý¢}
v¢s
 
inputsÿÿÿÿÿÿÿÿÿ
K¢H
"
states/0ÿÿÿÿÿÿÿÿÿn
"
states/1ÿÿÿÿÿÿÿÿÿn
p 
ª "s¢p
i¢f

0/0ÿÿÿÿÿÿÿÿÿn
EB

0/1/0ÿÿÿÿÿÿÿÿÿn

0/1/1ÿÿÿÿÿÿÿÿÿn
 
+__inference_lstm_cell_6_layer_call_fn_63533í¢}
v¢s
 
inputsÿÿÿÿÿÿÿÿÿ
K¢H
"
states/0ÿÿÿÿÿÿÿÿÿn
"
states/1ÿÿÿÿÿÿÿÿÿn
p
ª "c¢`

0ÿÿÿÿÿÿÿÿÿn
A>

1/0ÿÿÿÿÿÿÿÿÿn

1/1ÿÿÿÿÿÿÿÿÿn
+__inference_lstm_cell_6_layer_call_fn_63550í¢}
v¢s
 
inputsÿÿÿÿÿÿÿÿÿ
K¢H
"
states/0ÿÿÿÿÿÿÿÿÿn
"
states/1ÿÿÿÿÿÿÿÿÿn
p 
ª "c¢`

0ÿÿÿÿÿÿÿÿÿn
A>

1/0ÿÿÿÿÿÿÿÿÿn

1/1ÿÿÿÿÿÿÿÿÿn¼
G__inference_sequential_6_layer_call_and_return_conditional_losses_62323qA¢>
7¢4
*'
lstm_6_inputÿÿÿÿÿÿÿÿÿ
p

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 ¼
G__inference_sequential_6_layer_call_and_return_conditional_losses_62339qA¢>
7¢4
*'
lstm_6_inputÿÿÿÿÿÿÿÿÿ
p 

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 ¶
G__inference_sequential_6_layer_call_and_return_conditional_losses_62586k;¢8
1¢.
$!
inputsÿÿÿÿÿÿÿÿÿ
p

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 ¶
G__inference_sequential_6_layer_call_and_return_conditional_losses_62745k;¢8
1¢.
$!
inputsÿÿÿÿÿÿÿÿÿ
p 

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 
,__inference_sequential_6_layer_call_fn_62371dA¢>
7¢4
*'
lstm_6_inputÿÿÿÿÿÿÿÿÿ
p

 
ª "ÿÿÿÿÿÿÿÿÿ
,__inference_sequential_6_layer_call_fn_62402dA¢>
7¢4
*'
lstm_6_inputÿÿÿÿÿÿÿÿÿ
p 

 
ª "ÿÿÿÿÿÿÿÿÿ
,__inference_sequential_6_layer_call_fn_62760^;¢8
1¢.
$!
inputsÿÿÿÿÿÿÿÿÿ
p

 
ª "ÿÿÿÿÿÿÿÿÿ
,__inference_sequential_6_layer_call_fn_62775^;¢8
1¢.
$!
inputsÿÿÿÿÿÿÿÿÿ
p 

 
ª "ÿÿÿÿÿÿÿÿÿ­
#__inference_signature_wrapper_62427I¢F
¢ 
?ª<
:
lstm_6_input*'
lstm_6_inputÿÿÿÿÿÿÿÿÿ"1ª.
,
dense_6!
dense_6ÿÿÿÿÿÿÿÿÿ