// Tiny Lisp Interpreter in TypeScript

// ============ Types ============
type LispSymbol = { type: "symbol"; name: string };
type LispString = { type: "string"; value: string };
type LispFunction = {
  type: "lambda";
  params: string[];
  body: LispValue;
  env: Environment;
};
type LispValue = number | LispSymbol | LispString | boolean | LispValue[] | LispFunction | null;

class Environment {
  private vars: Map<string, LispValue> = new Map();
  private parent: Environment | null;

  constructor(parent: Environment | null = null) {
    this.parent = parent;
  }

  get(name: string): LispValue {
    if (this.vars.has(name)) {
      return this.vars.get(name)!;
    }
    if (this.parent) {
      return this.parent.get(name);
    }
    throw new Error(`Undefined variable: ${name}`);
  }

  set(name: string, value: LispValue): void {
    this.vars.set(name, value);
  }
}

// ============ Tokenizer ============
function tokenize(input: string): string[] {
  const tokens: string[] = [];
  let i = 0;

  while (i < input.length) {
    const char = input[i];

    // Skip whitespace
    if (/\s/.test(char)) {
      i++;
      continue;
    }

    // Skip comments (lines starting with ;)
    if (char === ";") {
      while (i < input.length && input[i] !== "\n") {
        i++;
      }
      continue;
    }

    // Parentheses
    if (char === "(" || char === ")") {
      tokens.push(char);
      i++;
      continue;
    }

    // Strings
    if (char === '"') {
      let str = '"';
      i++;
      while (i < input.length && input[i] !== '"') {
        if (input[i] === "\\") {
          str += input[i++];
        }
        str += input[i++];
      }
      str += '"';
      i++; // skip closing quote
      tokens.push(str);
      continue;
    }

    // Numbers and symbols
    let token = "";
    while (i < input.length && !/[\s()]/.test(input[i])) {
      token += input[i++];
    }
    if (token) {
      tokens.push(token);
    }
  }

  return tokens;
}

// ============ Parser ============
function parse(tokens: string[]): LispValue {
  if (tokens.length === 0) {
    throw new Error("Unexpected EOF");
  }

  const token = tokens.shift()!;

  if (token === "(") {
    const list: LispValue[] = [];
    while (tokens[0] !== ")") {
      if (tokens.length === 0) {
        throw new Error("Missing closing parenthesis");
      }
      list.push(parse(tokens));
    }
    tokens.shift(); // remove ')'
    return list;
  }

  if (token === ")") {
    throw new Error("Unexpected )");
  }

  // Atom: number, string, or symbol
  return parseAtom(token);
}

function parseAtom(token: string): LispValue {
  // Number
  if (/^-?\d+(\.\d+)?$/.test(token)) {
    return parseFloat(token);
  }

  // String (remove quotes)
  if (token.startsWith('"') && token.endsWith('"')) {
    return {
      type: "string",
      value: token.slice(1, -1).replace(/\\"/g, '"').replace(/\\\\/g, "\\"),
    };
  }

  // Boolean
  if (token === "#t" || token === "true") return true;
  if (token === "#f" || token === "false") return false;

  // Symbol (will be looked up in environment)
  return { type: "symbol", name: token };
}

function parseProgram(input: string): LispValue[] {
  const tokens = tokenize(input);
  const expressions: LispValue[] = [];

  while (tokens.length > 0) {
    expressions.push(parse(tokens));
  }

  return expressions;
}

// ============ Evaluator ============
function isSymbol(value: LispValue): value is LispSymbol {
  return (
    value !== null &&
    typeof value === "object" &&
    !Array.isArray(value) &&
    (value as any).type === "symbol"
  );
}

function isString(value: LispValue): value is LispString {
  return (
    value !== null &&
    typeof value === "object" &&
    !Array.isArray(value) &&
    (value as any).type === "string"
  );
}

function evaluate(expr: LispValue, env: Environment): LispValue {
  // Numbers, booleans, and null evaluate to themselves
  if (typeof expr === "number" || typeof expr === "boolean" || expr === null) {
    return expr;
  }

  // String literals evaluate to themselves
  if (isString(expr)) {
    return expr;
  }

  // Symbols are looked up in environment
  if (isSymbol(expr)) {
    return env.get(expr.name);
  }

  // Lambda functions evaluate to themselves
  if (isLambda(expr)) {
    return expr;
  }

  // Lists are function calls or special forms
  if (Array.isArray(expr)) {
    if (expr.length === 0) {
      return null;
    }

    const [first, ...rest] = expr;

    const firstSymbol = isSymbol(first) ? first.name : null;

    // Special forms
    if (firstSymbol === "define") {
      const [name, valueExpr] = rest;
      if (!isSymbol(name)) {
        throw new Error("define requires a symbol name");
      }
      const value = evaluate(valueExpr, env);
      env.set(name.name, value);
      return value;
    }

    if (firstSymbol === "if") {
      const [condition, consequent, alternate] = rest;
      const test = evaluate(condition, env);
      if (test !== false && test !== null && test !== 0) {
        return evaluate(consequent, env);
      } else {
        return alternate !== undefined ? evaluate(alternate, env) : null;
      }
    }

    if (firstSymbol === "lambda") {
      const [params, body] = rest;
      if (!Array.isArray(params)) {
        throw new Error("lambda requires a parameter list");
      }
      return {
        type: "lambda",
        params: (params as LispSymbol[]).map((p) => p.name),
        body,
        env,
      };
    }

    if (firstSymbol === "quote") {
      return rest[0];
    }

    // Function call
    const fn = evaluate(first, env);
    const args = rest.map((arg) => evaluate(arg, env));

    // Built-in functions
    if (typeof fn === "function") {
      return (fn as Function)(...args);
    }

    // Lambda call
    if (isLambda(fn)) {
      const newEnv = new Environment(fn.env);
      fn.params.forEach((param, i) => {
        newEnv.set(param, args[i]);
      });
      return evaluate(fn.body, newEnv);
    }

    throw new Error(`Cannot call ${JSON.stringify(fn)}`);
  }

  throw new Error(`Unknown expression type: ${JSON.stringify(expr)}`);
}

function isLambda(value: LispValue): value is LispFunction {
  return (
    value !== null &&
    typeof value === "object" &&
    !Array.isArray(value) &&
    (value as any).type === "lambda"
  );
}

// ============ Standard Environment ============
function createGlobalEnv(): Environment {
  const env = new Environment();

  // Arithmetic
  env.set("+", (...args: LispValue[]) =>
    (args as number[]).reduce((a, b) => a + b, 0)
  );
  env.set("-", (...args: LispValue[]) => {
    const nums = args as number[];
    if (nums.length === 1) return -nums[0];
    return nums.reduce((a, b) => a - b);
  });
  env.set("*", (...args: LispValue[]) =>
    (args as number[]).reduce((a, b) => a * b, 1)
  );
  env.set("/", (...args: LispValue[]) => {
    const nums = args as number[];
    return nums.reduce((a, b) => a / b);
  });

  // Comparison
  env.set("<", (a: LispValue, b: LispValue) => (a as number) < (b as number));
  env.set(">", (a: LispValue, b: LispValue) => (a as number) > (b as number));
  env.set("=", (a: LispValue, b: LispValue) => a === b);
  env.set("<=", (a: LispValue, b: LispValue) => (a as number) <= (b as number));
  env.set(">=", (a: LispValue, b: LispValue) => (a as number) >= (b as number));

  // List operations
  env.set("list", (...args: LispValue[]) => args);
  env.set("car", (list: LispValue) => {
    if (!Array.isArray(list) || list.length === 0) {
      throw new Error("car requires a non-empty list");
    }
    return list[0];
  });
  env.set("cdr", (list: LispValue) => {
    if (!Array.isArray(list)) {
      throw new Error("cdr requires a list");
    }
    return list.slice(1);
  });
  env.set("cons", (item: LispValue, list: LispValue) => {
    if (!Array.isArray(list)) {
      throw new Error("cons requires a list as second argument");
    }
    return [item, ...list];
  });
  env.set("null?", (list: LispValue) => Array.isArray(list) && list.length === 0);
  env.set("length", (list: LispValue) => {
    if (!Array.isArray(list)) {
      throw new Error("length requires a list");
    }
    return list.length;
  });

  // I/O
  env.set("print", (value: LispValue) => {
    console.log(formatValue(value));
    return value;
  });

  // Boolean operations
  env.set("not", (value: LispValue) => !value);
  env.set("and", (...args: LispValue[]) => args.every(Boolean));
  env.set("or", (...args: LispValue[]) => args.some(Boolean));

  return env;
}

function formatValue(value: LispValue): string {
  if (value === null) return "nil";
  if (value === true) return "#t";
  if (value === false) return "#f";
  if (typeof value === "number") return String(value);
  if (isString(value)) return value.value;
  if (isSymbol(value)) return value.name;
  if (Array.isArray(value)) {
    return "(" + value.map(formatValue).join(" ") + ")";
  }
  if (isLambda(value)) {
    return "<lambda>";
  }
  return String(value);
}

// ============ REPL ============
async function repl(): Promise<void> {
  const env = createGlobalEnv();

  const prompt = "lisp> ";
  process.stdout.write(prompt);

  for await (const line of console) {
    if (line.trim()) {
      try {
        const expressions = parseProgram(line);
        for (const expr of expressions) {
          const result = evaluate(expr, env);
          console.log(formatValue(result));
        }
      } catch (e) {
        console.error("Error:", (e as Error).message);
      }
    }
    process.stdout.write(prompt);
  }
}

async function runFile(filename: string): Promise<void> {
  const env = createGlobalEnv();
  const content = await Bun.file(filename).text();
  const expressions = parseProgram(content);

  for (const expr of expressions) {
    evaluate(expr, env);
  }
}

// ============ Main ============
const args = process.argv.slice(2);

if (args.length > 0) {
  runFile(args[0]);
} else {
  repl();
}
