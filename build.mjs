// Assembles the single-file app: node build.mjs → index.html
// Sources live in src/ so engine, species and app stay separate while editing.
import { readFileSync, writeFileSync } from 'node:fs';

const src = (f) => readFileSync(new URL(`./src/${f}`, import.meta.url), 'utf8');
const engine = ['engine-core.js', 'engine-sim.js', 'engine-creature.js', 'engine-render.js'].map(src).join('\n');
const html = `${src('shell.html')}
<!-- ===== ENGINE (species-independent) ===== -->
<script id="engine-src">
${engine}
</script>
<!-- ===== SPECIES DEFINITIONS ===== -->
<script id="species-src">
${src('species.js')}
</script>
<!-- ===== APPLICATION ===== -->
<script id="app-src">
${src('app.js')}
</script>
</body>
</html>
`;
writeFileSync(new URL('./index.html', import.meta.url), html);
console.log(`index.html written (${(html.length / 1024).toFixed(0)} KB)`);
