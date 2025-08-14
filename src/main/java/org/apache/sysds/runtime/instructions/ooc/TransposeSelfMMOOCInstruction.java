/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

package org.apache.sysds.runtime.instructions.ooc;

import org.apache.sysds.common.Opcodes;
import org.apache.sysds.conf.ConfigurationManager;
import org.apache.sysds.lops.MMTSJ;
import org.apache.sysds.lops.MMTSJ.MMTSJType;
import org.apache.sysds.runtime.DMLRuntimeException;
import org.apache.sysds.runtime.controlprogram.caching.MatrixObject;
import org.apache.sysds.runtime.controlprogram.context.ExecutionContext;
import org.apache.sysds.runtime.controlprogram.parfor.LocalTaskQueue;
import org.apache.sysds.runtime.functionobjects.Multiply;
import org.apache.sysds.runtime.functionobjects.Plus;
import org.apache.sysds.runtime.instructions.InstructionUtils;
import org.apache.sysds.runtime.instructions.cp.CPOperand;
import org.apache.sysds.runtime.instructions.cp.MMTSJCPInstruction;
import org.apache.sysds.runtime.instructions.spark.data.IndexedMatrixValue;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;
import org.apache.sysds.runtime.matrix.data.MatrixIndexes;
import org.apache.sysds.runtime.matrix.operators.AggregateBinaryOperator;
import org.apache.sysds.runtime.matrix.operators.AggregateOperator;
import org.apache.sysds.runtime.matrix.operators.BinaryOperator;
import org.apache.sysds.runtime.matrix.operators.Operator;
import org.apache.sysds.runtime.util.CommonThreadPool;

import java.util.HashMap;
import java.util.Map;
import java.util.concurrent.ExecutorService;

public class TransposeSelfMMOOCInstruction extends ComputationOOCInstruction {

	private final MMTSJType _tstype;

	protected TransposeSelfMMOOCInstruction(OOCType type, Operator op, CPOperand in1, CPOperand out, MMTSJType tstype, String opcode, String istr) {
		super(type, op, in1, out, opcode, istr);
		_tstype = tstype;
	}

	public static TransposeSelfMMOOCInstruction parseInstruction(String str) {
		String[] parts = InstructionUtils.getInstructionPartsWithValueType(str);
		InstructionUtils.checkNumFields(parts, 3);
		String opcode = parts[0];
		CPOperand in1 = new CPOperand(parts[1]); // the larget matrix (streamed)
		CPOperand out = new CPOperand(parts[2]);
		MMTSJType tstype = MMTSJType.valueOf(parts[3]);

		AggregateOperator agg = new AggregateOperator(0, Plus.getPlusFnObject());
		AggregateBinaryOperator ba = new AggregateBinaryOperator(Multiply.getMultiplyFnObject(), agg);

		return new TransposeSelfMMOOCInstruction(OOCType.TSMM, ba, in1, out, tstype, opcode, str);
	}

	@Override
	public void processInstruction( ExecutionContext ec ) {
		// 1. Identify the inputs
		MatrixObject min = ec.getMatrixObject(input1);
		long cols = min.getNumColumns();

		LocalTaskQueue<IndexedMatrixValue> qIn = min.getStreamHandle();
		LocalTaskQueue<IndexedMatrixValue> qOut = new LocalTaskQueue<>();
		BinaryOperator plus = InstructionUtils.parseBinaryOperator(Opcodes.PLUS.toString());
		ec.getMatrixObject(output).setStreamHandle(qOut);

//		MatrixBlock result = new MatrixBlock((int)cols, (int)cols, false);

		ExecutorService pool = CommonThreadPool.get();
		try {
			// Core logic: background thread
			pool.submit(() -> {
				IndexedMatrixValue tmp = null;
				try {
					HashMap<Long, MatrixBlock> partialResults = new HashMap<>();
					while ((tmp = qIn.dequeueTask()) != LocalTaskQueue.NO_MORE_TASKS) {
						MatrixBlock matrixBlock = (MatrixBlock) tmp.getValue();
						long rowIndex = tmp.getIndexes().getRowIndex();
//						long colIndex = tmp.getIndexes().getColumnIndex();

						MatrixBlock partialResult = matrixBlock.transposeSelfMatrixMultOperations(new MatrixBlock(), _tstype);


						// for single column block
						if ( min.getNumColumns() > min.getBlocksize() ) {
							qOut.enqueueTask(new IndexedMatrixValue(tmp.getIndexes(), partialResult));
						} else {
							MatrixBlock currAgg = partialResults.get(rowIndex);
							if (currAgg == null) {
								partialResults.put(rowIndex, partialResult);
							} else {
								currAgg.binaryOperationsInPlace(plus, partialResult);
							}

						}

						// emit aggregated blocks
						if( min.getNumColumns() > min.getBlocksize() ) {
							for (Map.Entry<Long, MatrixBlock> entry : partialResults.entrySet()) {
								MatrixIndexes outIndexes = new MatrixIndexes(entry.getKey(), 1L);
								qOut.enqueueTask(new IndexedMatrixValue(outIndexes, entry.getValue()));
							}
						}

					}
				}
				catch(Exception ex) {
					throw new DMLRuntimeException(ex);
				}
				finally {
					qOut.closeInput();
				}
			});
		} catch (Exception e) {
			throw new DMLRuntimeException(e);
		}
		finally {
			pool.shutdown();
		}
	}
}
