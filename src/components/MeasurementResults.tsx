"use client";

interface Measurements {
  [key: string]: number;
}

interface MeasurementResultsProps {
  measurements: Measurements;
}

const CIRCUMFERENCE_LABELS: Record<string, string> = {
  neck: "Cou",
  chest: "Poitrine",
  underbust: "Sous-poitrine",
  waist: "Taille",
  belly: "Ventre",
  hips: "Hanches",
  thigh: "Cuisse",
  knee: "Genou",
  calf: "Mollet",
  biceps: "Biceps",
  bicep: "Biceps",
};

const WIDTH_LABELS: Record<string, string> = {
  shoulder_width: "Largeur epaules",
  hip_width: "Largeur hanches",
};

const LENGTH_LABELS: Record<string, string> = {
  arm_length: "Longueur bras",
  forearm_length: "Avant-bras",
  leg_length: "Longueur jambe",
  inseam: "Entrejambe",
  torso_length: "Longueur torse",
  back_length: "Longueur dos",
};

function MeasurementRow({
  label,
  value,
}: {
  label: string;
  value: number;
}) {
  return (
    <div className="flex items-center justify-between py-2.5 px-3 rounded-lg hover:bg-[var(--surface-light)] transition-colors">
      <span className="text-[var(--foreground)]/80 text-sm">{label}</span>
      <span className="font-mono font-semibold text-[var(--accent)]">
        {value.toFixed(1)} <span className="text-xs text-[var(--foreground)]/50">cm</span>
      </span>
    </div>
  );
}

function MeasurementSection({
  title,
  icon,
  labels,
  measurements,
}: {
  title: string;
  icon: string;
  labels: Record<string, string>;
  measurements: Measurements;
}) {
  const items = Object.entries(labels).filter(([key]) => key in measurements);
  if (items.length === 0) return null;

  return (
    <div className="glass rounded-xl p-4">
      <h3 className="text-sm font-semibold text-[var(--primary-light)] uppercase tracking-wider mb-2 flex items-center gap-2">
        <span>{icon}</span> {title}
      </h3>
      <div className="space-y-0.5">
        {items.map(([key, label]) => (
          <MeasurementRow key={key} label={label} value={measurements[key]} />
        ))}
      </div>
    </div>
  );
}

function detectMorphology(m: Measurements): { type: string; letter: string; description: string } | null {
  const chest = m.chest;
  const waist = m.waist;
  const hips = m.hips;

  if (!chest || !waist || !hips) return null;

  // Ratios based on circumferences only
  const waistHipRatio = waist / hips;
  const waistChestRatio = waist / chest;

  const chestHipRatio = chest / hips;

  // X / Sablier : poitrine et hanches proches, taille bien marquee
  if (chestHipRatio >= 0.9 && chestHipRatio <= 1.1 && waistHipRatio < 0.75) {
    return { type: "Sablier", letter: "X", description: "Poitrine et hanches equilibrees, taille marquee" };
  }

  // V / Triangle inverse : poitrine nettement plus large que hanches
  if (chestHipRatio > 1.1) {
    return { type: "Triangle inverse", letter: "V", description: "Poitrine plus large que les hanches" };
  }

  // A / Triangle : hanches nettement plus larges que poitrine (ratio < 0.80)
  if (chestHipRatio < 0.80) {
    return { type: "Triangle", letter: "A", description: "Hanches plus larges que la poitrine" };
  }

  // O / Ovale : taille proche de la poitrine et des hanches
  if (waistHipRatio > 0.85 && waistChestRatio > 0.85) {
    return { type: "Ovale", letter: "O", description: "Silhouette arrondie, taille peu marquee" };
  }

  // H / Rectangle : proportions equilibrees
  if (chestHipRatio >= 0.80 && chestHipRatio < 0.90 && waistHipRatio >= 0.75) {
    return { type: "Rectangle", letter: "H", description: "Proportions droites, silhouette equilibree" };
  }

  // X par defaut si taille marquee
  if (waistHipRatio < 0.75) {
    return { type: "Sablier", letter: "X", description: "Poitrine et hanches equilibrees, taille marquee" };
  }

  return { type: "Rectangle", letter: "H", description: "Proportions droites, silhouette equilibree" };
}

export default function MeasurementResults({
  measurements,
}: MeasurementResultsProps) {
  const morphology = detectMorphology(measurements);

  return (
    <div className="space-y-4 w-full max-w-md mx-auto">
      <h2 className="text-xl font-bold text-center">Vos Mensurations</h2>

      {morphology && (
        <div className="glass rounded-xl p-4 text-center">
          <div className="text-4xl font-bold text-[var(--primary-light)] mb-1">
            {morphology.letter}
          </div>
          <div className="text-sm font-semibold text-[var(--foreground)]">
            Morphologie {morphology.type}
          </div>
          <div className="text-xs text-[var(--foreground)]/50 mt-1">
            {morphology.description}
          </div>
        </div>
      )}

      <MeasurementSection
        title="Circonferences"
        icon="O"
        labels={CIRCUMFERENCE_LABELS}
        measurements={measurements}
      />

      <MeasurementSection
        title="Largeurs"
        icon="+"
        labels={WIDTH_LABELS}
        measurements={measurements}
      />

      <MeasurementSection
        title="Longueurs"
        icon="|"
        labels={LENGTH_LABELS}
        measurements={measurements}
      />
    </div>
  );
}
