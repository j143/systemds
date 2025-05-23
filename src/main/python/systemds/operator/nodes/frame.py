# -------------------------------------------------------------
#
# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
#
# -------------------------------------------------------------

__all__ = ["Frame"]

import os
from typing import TYPE_CHECKING, Dict, Iterable, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
from py4j.java_gateway import JavaObject, JVMView
from systemds.operator.operation_node import OperationNode
from systemds.operator.nodes.multi_return import MultiReturn
from systemds.operator.nodes.scalar import Scalar
from systemds.operator.nodes.matrix import Matrix
from systemds.utils.consts import VALID_INPUT_TYPES
from systemds.utils.converters import frame_block_to_pandas, pandas_to_frame_block
from systemds.utils.helpers import (
    check_is_empty_slice,
    check_no_less_than_zero,
    get_slice_string,
)

if TYPE_CHECKING:
    # to avoid cyclic dependencies during runtime
    from systemds.context import SystemDSContext


def to_frame(self):
    return Frame(self.sds_context, "as.frame", [self])


OperationNode.to_frame = to_frame


class Frame(OperationNode):
    _pd_dataframe: pd.DataFrame

    def __init__(
        self,
        sds_context: "SystemDSContext",
        operation: str,
        unnamed_input_nodes: Union[str, Iterable[VALID_INPUT_TYPES]] = None,
        named_input_nodes: Dict[str, VALID_INPUT_TYPES] = None,
        local_data: pd.DataFrame = None,
        brackets: bool = False,
    ) -> "Frame":
        is_python_local_data = False
        if local_data is not None:
            self._pd_dataframe = local_data
            is_python_local_data = True
        else:
            self._pd_dataframe = None

        super().__init__(
            sds_context,
            operation,
            unnamed_input_nodes,
            named_input_nodes,
            is_python_local_data,
            brackets,
            is_datatype_none=False,
        )

    def pass_python_data_to_prepared_script(
        self, sds, var_name: str, prepared_script: JavaObject
    ) -> None:
        assert (
            self.is_python_local_data
        ), "Can only pass data to prepared script if it is python local!"
        if self._is_pandas():
            prepared_script.setFrame(
                var_name, pandas_to_frame_block(sds, self._pd_dataframe), True
            )  # True for reuse

    def code_line(
        self,
        var_name: str,
        unnamed_input_vars: Sequence[str],
        named_input_vars: Dict[str, str],
    ) -> str:
        code_line = super().code_line(var_name, unnamed_input_vars, named_input_vars)
        if self._is_pandas():
            code_line = code_line.format(file_name=var_name)
        return code_line

    def compute(self, verbose: bool = False, lineage: bool = False) -> pd.DataFrame:
        if self._is_pandas():
            self.sds_context._log.info("Pandas Frame - No Compilation necessary")
            return self._pd_dataframe
        else:
            return super().compute(verbose, lineage)

    def _parse_output_result_variables(self, result_variables):
        return frame_block_to_pandas(
            self.sds_context,
            result_variables.getFrameBlock(self._script.out_var_name[0]),
        )

    def _is_pandas(self) -> bool:
        return self._pd_dataframe is not None

    def transform_encode(self, spec: "Scalar"):
        params_dict = {"target": self, "spec": spec}

        frame = Frame(self.sds_context, "")
        matrix = Matrix(self.sds_context, "")
        output_nodes = [matrix, frame]

        op = MultiReturn(
            self.sds_context,
            "transformencode",
            output_nodes,
            named_input_nodes=params_dict,
        )

        frame._unnamed_input_nodes = [op]
        matrix._unnamed_input_nodes = [op]

        return op

    def transform_apply(self, spec: "Scalar", meta: "Frame"):
        params_dict = {"target": self, "spec": spec, "meta": meta}
        return Matrix(self.sds_context, "transformapply", named_input_nodes=params_dict)

    def rbind(self, other) -> "Frame":
        """
        Row-wise frame concatenation, by concatenating the second frame as additional rows to the first frame.
        :param: The other frame to bind to the right hand side
        :return: The OperationNode containing the concatenated frames.
        """

        return Frame(self.sds_context, "rbind", [self, other])

    def cbind(self, other) -> "Frame":
        """
        Column-wise frame concatenation, by concatenating the second frame as additional columns to the first frame.
        :param: The other frame to bind to the right hand side.
        :return: The Frame containing the concatenated frames.
        """
        return Frame(self.sds_context, "cbind", [self, other])

    def replace(self, pattern: str, replacement: str) -> "Frame":
        """
        Replace all instances of string with replacement string
        :param: pattern the string to replace
        :param: replacement the string to replace with
        :return: The Frame containing the replaced values
        """
        return Frame(
            self.sds_context,
            "replace",
            named_input_nodes={
                "target": self,
                "pattern": f"'{pattern}'",
                "replacement": f"'{replacement}'",
            },
        )

    def to_string(self, **kwargs: Dict[str, VALID_INPUT_TYPES]) -> "Scalar":
        """Converts the input to a string representation.
        :return: `Scalar` containing the string.
        """
        return Scalar(self.sds_context, "toString", [self], kwargs)

    def __str__(self):
        return "FrameNode"

    def nRow(self) -> "Scalar":
        return Scalar(self.sds_context, "nrow", [self])

    def nCol(self) -> "Scalar":
        return Scalar(self.sds_context, "ncol", [self])

    def __getitem__(self, i) -> "Frame":
        if isinstance(i, tuple) and len(i) > 2:
            raise ValueError("Maximum of two dimensions are allowed")
        elif isinstance(i, list):
            check_no_less_than_zero(i)
            slice = self.sds_context.from_numpy(np.array(i)) + 1
            select = Matrix(self.sds_context, "table", [slice, 1, self.nRow(), 1])
            ret = Frame(
                self.sds_context,
                "removeEmpty",
                [],
                {"target": self, "margin": '"rows"', "select": select},
            )
            return ret
        elif isinstance(i, tuple) and isinstance(i[0], list) and isinstance(i[1], list):
            raise NotImplementedError("double slicing is not supported yet")
        elif (
            isinstance(i, tuple)
            and check_is_empty_slice(i[0])
            and isinstance(i[1], list)
        ):
            check_no_less_than_zero(i[1])
            slice = self.sds_context.from_numpy(np.array(i[1])) + 1
            select = Matrix(self.sds_context, "table", [slice, 1, self.nCol(), 1])
            ret = Frame(
                self.sds_context,
                "removeEmpty",
                [],
                {"target": self, "margin": '"cols"', "select": select},
            )
            return ret
        else:
            sliceIns = get_slice_string(i)
            return Frame(self.sds_context, "", [self, sliceIns], brackets=True)
