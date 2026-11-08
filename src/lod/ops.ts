export interface Ops<S, B> {
  constant(value: number): S;
  add(a: S, b: S): S;
  subtract(a: S, b: S): S;
  multiply(a: S, b: S): S;
  divide(a: S, b: S): S;
  minimum(a: S, b: S): S;
  maximum(a: S, b: S): S;
  floor(a: S): S;
  sqrt(a: S): S;
  log2(a: S): S;
  power2(a: S): S;
  below(a: S, b: S): B;
  both(a: B, b: B): B;
}

export const numberOps: Ops<number, boolean> = {
  constant: (value) => value,
  add: (a, b) => a + b,
  subtract: (a, b) => a - b,
  multiply: (a, b) => a * b,
  divide: (a, b) => a / b,
  minimum: Math.min,
  maximum: Math.max,
  floor: Math.floor,
  sqrt: Math.sqrt,
  log2: Math.log2,
  power2: (a) => 2 ** a,
  below: (a, b) => a < b,
  both: (a, b) => a && b,
};
