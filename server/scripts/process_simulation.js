const fs = require('fs');
const Module = require('module');
const path = require('path');

// Path defaults match the parent directory holding the Delineo sibling repos.
// Override DELINEO_ROOT or DELINEO_FULLSTACK_WORKTREE for other machines.
const root = process.env.DELINEO_ROOT || '/Users/ryad/Code/delineo';
const fullstackWorktree = process.env.DELINEO_FULLSTACK_WORKTREE
  || path.join(root, '_worktrees', 'fullstack-perf-phase1');
const fullstackNodeModules = process.env.DELINEO_FULLSTACK_NODE_MODULES
  || path.join(root, 'Fullstack', 'node_modules');
process.env.NODE_PATH = [
  fullstackNodeModules,
  process.env.NODE_PATH || ''
].filter(Boolean).join(path.delimiter);
Module._initPaths();

const ts = require('typescript');

require.extensions['.ts'] = function loadTs(module, filename) {
  const source = fs.readFileSync(filename, 'utf8');
  const output = ts.transpileModule(source, {
    compilerOptions: {
      module: ts.ModuleKind.CommonJS,
      target: ts.ScriptTarget.ES2020,
      esModuleInterop: true,
      moduleResolution: ts.ModuleResolutionKind.Node10,
      skipLibCheck: true
    },
    fileName: filename
  }).outputText;
  module._compile(output, filename);
};

function readArgs(argv) {
  const args = {};
  for (let i = 2; i < argv.length; i += 2) {
    const key = argv[i];
    const value = argv[i + 1];
    if (!key || !key.startsWith('--') || value === undefined) {
      throw new Error(`Invalid argument list near ${key || '<end>'}`);
    }
    args[key.slice(2)] = value;
  }
  return args;
}

async function main() {
  const args = readArgs(process.argv);
  const { processSimulation } = require(path.join(
    fullstackWorktree,
    'src',
    'lib',
    'sim-processor.ts'
  ));

  await processSimulation({
    simDataId: Number(args['sim-data-id'] || 1),
    simdataPath: args.simdata,
    patternsPath: args.patterns,
    papdataId: args['papdata-id'],
    mapCachePath: args['map-cache'],
    totalLength: Number(args.length)
  });
}

main().catch((error) => {
  console.error(error);
  process.exit(1);
});
