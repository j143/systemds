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

package org.apache.sysds.test.component.matrixmult;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;

import java.util.Arrays;
import java.util.Random;

import org.apache.sysds.runtime.matrix.data.LibMatrixMult;
import org.junit.Test;

/**
 * Performance and correctness tests for the five vector-primitive methods
 * in LibMatrixMult that were upgraded from scalar 8-block unrolling to the
 * Java Vector API (jdk.incubator.vector):
 *
 *  1. vectAdd(double[], double[], int, int, int)
 *  2. vectAdd4(double[], double[], double[], double[], double[], int, int, int)
 *  3. vectAddInPlace(double, double[], int, int)
 *  4. vectMultiplyInPlace(double, double[], int, int)
 *  5. vectSubtract (exercised indirectly via matrix operations that call it)
 *
 * Each test verifies correctness against a naive reference and measures
 * throughput so that the improvement can be observed in CI output.
 */
public class VectorPrimitivesPerformanceTest {

	// Vector length used for performance runs (must be large enough to benefit
	// from SIMD, but small enough that the test finishes quickly).
	private static final int LEN  = 1 << 20;  // 1 M doubles ≈ 8 MB
	private static final int WARMUP = 20;
	private static final int REP   = 50;

	// -----------------------------------------------------------------------
	// 1.  vectAdd  (dense c += a)
	// -----------------------------------------------------------------------
	@Test
	public void testVectAddCorrectness() {
		int len = 1025; // not a multiple of any SIMD lane count
		double[] a = rnd(len);
		double[] c = rnd(len);
		double[] expected = c.clone();
		for(int i = 0; i < len; i++) expected[i] += a[i];

		LibMatrixMult.vectAdd(a, c, 0, 0, len);
		assertArrayEquals("vectAdd correctness", expected, c, 1e-12);
	}

	@Test
	public void testVectAddWithOffsets() {
		int len = 512;
		int ai = 7, ci = 3;
		double[] a = rnd(ai + len);
		double[] c = rnd(ci + len);
		double[] expected = c.clone();
		for(int i = 0; i < len; i++) expected[ci + i] += a[ai + i];

		LibMatrixMult.vectAdd(a, c, ai, ci, len);
		assertArrayEquals("vectAdd with offsets", expected, c, 1e-12);
	}

	@Test
	public void testVectAddPerformance() {
		double[] a = rnd(LEN), c = new double[LEN];

		// warm-up
		for(int i = 0; i < WARMUP; i++) {
			Arrays.fill(c, 0.0);
			LibMatrixMult.vectAdd(a, c, 0, 0, LEN);
		}

		// measure
		Arrays.fill(c, 0.0);
		long t0 = System.nanoTime();
		for(int i = 0; i < REP; i++) {
			LibMatrixMult.vectAdd(a, c, 0, 0, LEN);
		}
		double ms = (System.nanoTime() - t0) / 1e6 / REP;
		System.out.printf("vectAdd         (%d doubles): %.3f ms/call%n", LEN, ms);
	}

	// -----------------------------------------------------------------------
	// 2.  vectAddInPlace  (c[i] += scalar)
	// -----------------------------------------------------------------------
	@Test
	public void testVectAddInPlaceCorrectness() {
		int len = 1025;
		double aval = 3.14;
		double[] c = rnd(len);
		double[] expected = c.clone();
		for(int i = 0; i < len; i++) expected[i] += aval;

		LibMatrixMult.vectAddInPlace(aval, c, 0, len);
		assertArrayEquals("vectAddInPlace correctness", expected, c, 1e-12);
	}

	@Test
	public void testVectAddInPlaceWithOffset() {
		int len = 512;
		int ci = 5;
		double aval = -2.7;
		double[] c = rnd(ci + len);
		double[] expected = c.clone();
		for(int i = ci; i < ci + len; i++) expected[i] += aval;

		LibMatrixMult.vectAddInPlace(aval, c, ci, len);
		assertArrayEquals("vectAddInPlace with offset", expected, c, 1e-12);
	}

	@Test
	public void testVectAddInPlacePerformance() {
		double[] c = rnd(LEN);
		double aval = 0.5;

		for(int i = 0; i < WARMUP; i++)
			LibMatrixMult.vectAddInPlace(aval, c, 0, LEN);

		long t0 = System.nanoTime();
		for(int i = 0; i < REP; i++)
			LibMatrixMult.vectAddInPlace(aval, c, 0, LEN);
		double ms = (System.nanoTime() - t0) / 1e6 / REP;
		System.out.printf("vectAddInPlace  (%d doubles): %.3f ms/call%n", LEN, ms);
	}

	// -----------------------------------------------------------------------
	// 3.  vectMultiplyInPlace  (c[i] *= scalar)
	// -----------------------------------------------------------------------
	@Test
	public void testVectMultiplyInPlaceCorrectness() {
		int len = 1025;
		double aval = 2.0;
		double[] c = rnd(len);
		double[] expected = c.clone();
		for(int i = 0; i < len; i++) expected[i] *= aval;

		LibMatrixMult.vectMultiplyInPlace(aval, c, 0, len);
		assertArrayEquals("vectMultiplyInPlace correctness", expected, c, 1e-12);
	}

	@Test
	public void testVectMultiplyInPlaceWithOffset() {
		int len = 512;
		int ci = 3;
		double aval = 0.5;
		double[] c = rnd(ci + len);
		double[] expected = c.clone();
		for(int i = ci; i < ci + len; i++) expected[i] *= aval;

		LibMatrixMult.vectMultiplyInPlace(aval, c, ci, len);
		assertArrayEquals("vectMultiplyInPlace with offset", expected, c, 1e-12);
	}

	@Test
	public void testVectMultiplyInPlacePerformance() {
		double[] c = rnd(LEN);
		// keep values near 1 so they don't blow up / underflow across reps
		double aval = 1.0;

		for(int i = 0; i < WARMUP; i++)
			LibMatrixMult.vectMultiplyInPlace(aval, c, 0, LEN);

		long t0 = System.nanoTime();
		for(int i = 0; i < REP; i++)
			LibMatrixMult.vectMultiplyInPlace(aval, c, 0, LEN);
		double ms = (System.nanoTime() - t0) / 1e6 / REP;
		System.out.printf("vectMultiplyInPlace (%d doubles): %.3f ms/call%n", LEN, ms);
	}

	// -----------------------------------------------------------------------
	// 4.  vectSubtract – exercised indirectly via WSLoss / column reduction
	//     paths that invoke it internally, so we test correctness by computing
	//     the equivalent operation and comparing.
	// -----------------------------------------------------------------------
	@Test
	public void testVectSubtractCorrectnessViaScalar() {
		// vectSubtract is package-private; test through reflection so the
		// production code path is exercised without changing visibility.
		int len = 1025;
		double[] a = rnd(len);
		double[] c = rnd(len);
		double[] expected = c.clone();
		for(int i = 0; i < len; i++) expected[i] -= a[i];

		invokeVectSubtract(a, c, 0, 0, len);
		assertArrayEquals("vectSubtract correctness", expected, c, 1e-12);
	}

	@Test
	public void testVectSubtractPerformance() {
		double[] a = rnd(LEN), c = rnd(LEN);

		for(int i = 0; i < WARMUP; i++) invokeVectSubtract(a, c, 0, 0, LEN);
		long t0 = System.nanoTime();
		for(int i = 0; i < REP; i++) invokeVectSubtract(a, c, 0, 0, LEN);
		double ms = (System.nanoTime() - t0) / 1e6 / REP;
		System.out.printf("vectSubtract    (%d doubles): %.3f ms/call%n", LEN, ms);
	}

	// -----------------------------------------------------------------------
	// 5.  vectAdd4 – also package-private; test via reflection
	// -----------------------------------------------------------------------
	@Test
	public void testVectAdd4Correctness() {
		int len = 1025;
		double[] a1 = rnd(len), a2 = rnd(len), a3 = rnd(len), a4 = rnd(len);
		double[] c  = rnd(len);
		double[] expected = c.clone();
		for(int i = 0; i < len; i++)
			expected[i] += a1[i] + a2[i] + a3[i] + a4[i];

		invokeVectAdd4(a1, a2, a3, a4, c, 0, 0, len);
		assertArrayEquals("vectAdd4 correctness", expected, c, 1e-12);
	}

	@Test
	public void testVectAdd4WithOffsets() {
		int len = 512;
		int ai = 4, ci = 2;
		double[] a1 = rnd(ai + len), a2 = rnd(ai + len),
		         a3 = rnd(ai + len), a4 = rnd(ai + len);
		// vectAdd4 uses the *same* offset ai for all four source arrays
		double[] a1s = Arrays.copyOfRange(a1, ai, ai + len);
		double[] a2s = Arrays.copyOfRange(a2, ai, ai + len);
		double[] a3s = Arrays.copyOfRange(a3, ai, ai + len);
		double[] a4s = Arrays.copyOfRange(a4, ai, ai + len);
		double[] c  = rnd(ci + len);
		double[] expected = c.clone();
		for(int i = 0; i < len; i++)
			expected[ci + i] += a1s[i] + a2s[i] + a3s[i] + a4s[i];

		invokeVectAdd4(a1, a2, a3, a4, c, ai, ci, len);
		assertArrayEquals("vectAdd4 with offsets", expected, c, 1e-12);
	}

	@Test
	public void testVectAdd4Performance() {
		double[] a1 = rnd(LEN), a2 = rnd(LEN), a3 = rnd(LEN), a4 = rnd(LEN);
		double[] c  = new double[LEN];

		for(int i = 0; i < WARMUP; i++) {
			Arrays.fill(c, 0.0);
			invokeVectAdd4(a1, a2, a3, a4, c, 0, 0, LEN);
		}

		Arrays.fill(c, 0.0);
		long t0 = System.nanoTime();
		for(int i = 0; i < REP; i++)
			invokeVectAdd4(a1, a2, a3, a4, c, 0, 0, LEN);
		double ms = (System.nanoTime() - t0) / 1e6 / REP;
		System.out.printf("vectAdd4        (%d doubles): %.3f ms/call%n", LEN, ms);
	}

	// -----------------------------------------------------------------------
	// helpers
	// -----------------------------------------------------------------------
	private static final Random RND = new Random(42);

	private static double[] rnd(int len) {
		double[] a = new double[len];
		for(int i = 0; i < len; i++) a[i] = RND.nextDouble() * 2 - 1;
		return a;
	}

	private static java.lang.reflect.Method vectAdd4Method;
	private static java.lang.reflect.Method vectSubtractMethod;

	private static void invokeVectSubtract(double[] a, double[] c, int ai, int ci, int len) {
		try {
			if(vectSubtractMethod == null) {
				vectSubtractMethod = LibMatrixMult.class.getDeclaredMethod(
					"vectSubtract", double[].class, double[].class, int.class, int.class, int.class);
				vectSubtractMethod.setAccessible(true);
			}
			vectSubtractMethod.invoke(null, a, c, ai, ci, len);
		} catch(Exception e) {
			throw new RuntimeException("Could not invoke vectSubtract", e);
		}
	}

	private static void invokeVectAdd4(double[] a1, double[] a2, double[] a3, double[] a4,
		double[] c, int ai, int ci, int len)
	{
		try {
			if(vectAdd4Method == null) {
				vectAdd4Method = LibMatrixMult.class.getDeclaredMethod(
					"vectAdd4",
					double[].class, double[].class, double[].class, double[].class,
					double[].class, int.class, int.class, int.class);
				vectAdd4Method.setAccessible(true);
			}
			vectAdd4Method.invoke(null, a1, a2, a3, a4, c, ai, ci, len);
		} catch(Exception e) {
			throw new RuntimeException("Could not invoke vectAdd4", e);
		}
	}
}
