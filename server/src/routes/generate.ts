import { Router, Response } from 'express';
import multer from 'multer';
import path from 'path';
import { fileURLToPath } from 'url';
import crypto from 'crypto';
import { pool } from '../db/pool.js';
import { generateUUID } from '../db/sqlite.js';
import { config } from '../config/index.js';
import { authMiddleware, AuthenticatedRequest } from '../middleware/auth.js';
import { getGradioClient } from '../services/gradio-client.js';
import { getSunoBearerToken, getSunoSettings, launchChromeCdp, parseCookieString, SUNO_CHROME_CDP_URL } from '../services/sunoAuth.js';
import {
  generateMusicViaAPI,
  getJobStatus,
  getAudioStream,
  discoverEndpoints,
  checkSpaceHealth,
  cleanupJob,
  getJobRawResponse,
  downloadAudioToBuffer,
  resolvePythonPath,
} from '../services/acestep.js';
import { getStorageProvider } from '../services/storage/factory.js';

const router = Router();

// Auto-generate a song title from lyrics or style when none is provided
function autoTitle(params: { title?: string; lyrics?: string; instrumental?: boolean; style?: string; songDescription?: string }): string {
  if (params.title?.trim()) return params.title.trim();

  // Try first meaningful lyric line (skip section markers like [verse], [chorus])
  if (!params.instrumental && params.lyrics) {
    for (const line of params.lyrics.split('\n')) {
      const t = line.trim();
      if (t && !/^\[.*\]$/.test(t)) {
        return t.length > 40 ? t.slice(0, 40).trimEnd() + '…' : t;
      }
    }
  }

  // Fall back to first 4 words of style or description
  const source = params.style || params.songDescription || '';
  if (source) {
    const words = source.trim().split(/\s+/).slice(0, 4).join(' ');
    return words.charAt(0).toUpperCase() + words.slice(1);
  }

  return 'Untitled';
}

const audioUpload = multer({
  storage: multer.memoryStorage(),
  limits: { fileSize: 25 * 1024 * 1024 }, // 25MB max
  fileFilter: (_req, file, cb) => {
    const allowedTypes = [
      'audio/mpeg',
      'audio/mp3', // Alternative MIME type for MP3
      'audio/mpeg3',
      'audio/x-mpeg-3',
      'audio/wav',
      'audio/x-wav',
      'audio/flac',
      'audio/x-flac',
      'audio/mp4',
      'audio/x-m4a',
      'audio/aac',
      'audio/ogg',
      'audio/webm',
      'video/mp4',
    ];

    // Also check file extension as fallback
    const allowedExtensions = ['.mp3', '.wav', '.flac', '.m4a', '.mp4', '.aac', '.ogg', '.webm', '.opus'];
    const fileExt = file.originalname.toLowerCase().match(/\.[^.]+$/)?.[0];

    if (allowedTypes.includes(file.mimetype) || (fileExt && allowedExtensions.includes(fileExt))) {
      cb(null, true);
    } else {
      cb(new Error(`Invalid file type. Only common audio formats are allowed. Received: ${file.mimetype} (${file.originalname})`));
    }
  }
});

interface GenerateBody {
  provider?: 'ace' | 'suno';

  // Mode
  customMode: boolean;

  // Simple Mode
  songDescription?: string;

  // Custom Mode
  lyrics: string;
  style: string;
  title: string;

  // Common
  instrumental: boolean;
  vocalLanguage?: string;

  // Music Parameters
  duration?: number;
  bpm?: number;
  keyScale?: string;
  timeSignature?: string;

  // Generation Settings
  inferenceSteps?: number;
  guidanceScale?: number;
  batchSize?: number;
  randomSeed?: boolean;
  seed?: number;
  thinking?: boolean;
  audioFormat?: 'mp3' | 'flac';
  inferMethod?: 'ode' | 'sde';
  shift?: number;

  // LM Parameters
  lmTemperature?: number;
  lmCfgScale?: number;
  lmTopK?: number;
  lmTopP?: number;
  lmNegativePrompt?: string;
  lmBackend?: 'pt' | 'vllm';
  lmModel?: string;

  // Expert Parameters
  referenceAudioUrl?: string;
  sourceAudioUrl?: string;
  referenceAudioTitle?: string;
  sourceAudioTitle?: string;
  audioCodes?: string;
  repaintingStart?: number;
  repaintingEnd?: number;
  instruction?: string;
  audioCoverStrength?: number;
  taskType?: string;
  useAdg?: boolean;
  cfgIntervalStart?: number;
  cfgIntervalEnd?: number;
  customTimesteps?: string;
  useCotMetas?: boolean;
  useCotCaption?: boolean;
  useCotLanguage?: boolean;
  autogen?: boolean;
  constrainedDecodingDebug?: boolean;
  allowLmBatch?: boolean;
  getScores?: boolean;
  getLrc?: boolean;
  scoreScale?: number;
  lmBatchChunkSize?: number;
  trackName?: string;
  completeTrackClasses?: string[];
  isFormatCaption?: boolean;

  // Model selection
  ditModel?: string;

  // Suno-compatible API
  sunoBaseUrl?: string;
  sunoApiKey?: string;
  sunoEndpoint?: string;
  sunoModel?: string;
  sunoProjectId?: string;
  sunoProjectName?: string;
  sunoWaitForAudio?: boolean;
  sunoPollSeconds?: number;
  sunoNegativeTags?: string;
  sunoTask?: string;
  sunoVocalGender?: 'm' | 'f' | '';
  sunoWeirdness?: number;
  sunoStyleInfluence?: number;
  sunoContinueClipId?: string;
  sunoContinueAt?: number;
  sunoCoverClipId?: string;
  sunoArtistClipId?: string;
  sunoInfillStartS?: number;
  sunoInfillEndS?: number;
  sunoStemTypeId?: string;
  sunoOverrideFieldsJson?: string;
  sunoBrowserSubmit?: boolean;
}

function buildSunoUrl(baseUrl: string, endpoint: string): string {
  const base = new URL(baseUrl);
  if (!['http:', 'https:'].includes(base.protocol)) {
    throw new Error('Suno Base URL must start with http:// or https://');
  }
  return new URL(endpoint || '/api/generate', base).toString();
}

async function buildSunoHeaders(apiKey?: string): Promise<Record<string, string>> {
  const settings = await getSunoSettings();
  const token = apiKey?.trim();
  const cookies = parseCookieString(settings.suno_cookie || '');
  const deviceId = cookies.ajs_anonymous_id || crypto.randomUUID();
  const headers: Record<string, string> = {
    'Content-Type': 'application/json',
    'Affiliate-Id': 'undefined',
    'Device-Id': `"${deviceId}"`,
    'x-suno-client': 'Android prerelease-4nt180t 1.0.42',
    'X-Requested-With': 'com.suno.android',
    'sec-ch-ua': '"Chromium";v="130", "Android WebView";v="130", "Not?A_Brand";v="99"',
    'sec-ch-ua-mobile': '?1',
    'sec-ch-ua-platform': '"Android"',
    'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/145.0.0.0 Safari/537.36',
    Origin: 'https://suno.com',
    Referer: 'https://suno.com/',
  };
  if (settings.suno_cookie?.trim()) {
    headers.Cookie = settings.suno_cookie.trim();
  }
  if (token) {
    headers.Authorization = `Bearer ${token}`;
    headers['x-api-key'] = token;
  }
  return headers;
}

function addOptionalString(target: Record<string, unknown>, key: string, value?: string): void {
  const trimmed = value?.trim();
  if (trimmed) target[key] = trimmed;
}

function addOptionalNumber(target: Record<string, unknown>, key: string, value?: number): void {
  if (typeof value === 'number' && Number.isFinite(value)) {
    target[key] = value;
  }
}

function clampSunoSlider(value?: number): number | undefined {
  if (typeof value !== 'number' || !Number.isFinite(value)) return undefined;
  return Math.max(0, Math.min(100, value));
}

function parseOptionalJsonObject(value?: string): Record<string, unknown> | undefined {
  if (!value?.trim()) return undefined;
  const parsed = JSON.parse(value);
  if (!parsed || typeof parsed !== 'object' || Array.isArray(parsed)) {
    throw new Error('Override fields must be a JSON object');
  }
  return parsed as Record<string, unknown>;
}

function isManualSunoFallbackError(status: number, external: unknown): boolean {
  if (status === 401) return true;
  const text = typeof external === 'string' ? external : JSON.stringify(external || {});
  return (
    status === 422 && (
      text.includes('token_validation_failed') ||
      text.includes('captcha') ||
      text.includes("couldn't verify")
    )
  ) || (
    status === 403 && (
      text.includes('control sliders') ||
      text.includes('permission_denied') ||
      text.includes("current plan")
    )
  );
}

function escapeRegExp(value: string): string {
  return value.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
}

async function clickFirstVisible(locator: any, timeout = 3000): Promise<boolean> {
  const count = await locator.count().catch(() => 0);
  for (let index = 0; index < count; index += 1) {
    const item = locator.nth(index);
    if (await item.isVisible({ timeout: 500 }).catch(() => false)) {
      await item.click({ timeout });
      return true;
    }
  }
  return false;
}

async function closeSunoDialog(page: any): Promise<void> {
  await page.getByRole('button', { name: /^Close$/i }).click({ timeout: 1000 }).catch(() => undefined);
  await page.keyboard.press('Escape').catch(() => undefined);
  await page.waitForTimeout(250).catch(() => undefined);
}

async function selectSunoWorkspace(page: any, workspaceName?: string): Promise<boolean> {
  const name = workspaceName?.trim();
  if (!name) return false;
  const optionName = new RegExp(`^${escapeRegExp(name)}(?:\\s*\\(|$)`, 'i');
  let selected = false;

  await closeSunoDialog(page);

  const saveToButton = page.locator('button:visible').filter({ hasText: new RegExp(`^${escapeRegExp(name)}$`, 'i') }).last();
  if (await saveToButton.isVisible({ timeout: 2000 }).catch(() => false)) {
    selected = true;
  } else {
    const currentSaveTo = page.locator('button:visible').filter({ hasText: /^(My Workspace|Test)$/i }).last();
    if (await currentSaveTo.isVisible({ timeout: 2000 }).catch(() => false)) {
      await currentSaveTo.click({ timeout: 5000, force: true });
      await page.waitForTimeout(500);
      const optionLocators = [
        page.getByRole('option', { name: optionName }),
        page.getByRole('menuitem', { name: optionName }),
        page.locator('button:visible').filter({ hasText: optionName }).first(),
        page.locator('[role="dialog"], [role="listbox"], [role="menu"], [data-radix-popper-content-wrapper]').getByText(optionName),
      ];
      for (const option of optionLocators) {
        if (await clickFirstVisible(option, 3000)) {
          selected = true;
          break;
        }
      }
      await page.keyboard.press('Escape').catch(() => undefined);
    }
  }

  return selected;
}

async function setSunoSlider(page: any, label: string, value?: number): Promise<boolean> {
  const clamped = clampSunoSlider(value);
  if (typeof clamped !== 'number') return false;
  const slider = page.getByRole('slider', { name: new RegExp(`^${escapeRegExp(label)}$`, 'i') }).first();
  if (!await slider.isVisible({ timeout: 3000 }).catch(() => false)) return false;

  const box = await slider.boundingBox();
  if (!box) return false;
  await page.mouse.click(box.x + (box.width * clamped / 100), box.y + box.height / 2);
  await page.waitForTimeout(250);
  const upgradeDialog = page.getByText(/Available with Pro|Upgrade to create|Advanced Sliders/i).first();
  if (await upgradeDialog.isVisible({ timeout: 500 }).catch(() => false)) {
    await closeSunoDialog(page);
    return false;
  }
  return true;
}

async function runSunoBrowserGenerate(body: GenerateBody) {
  await launchChromeCdp();
  const { chromium } = await import('rebrowser-playwright-core');
  const browser = await chromium.connectOverCDP(SUNO_CHROME_CDP_URL);
  const context = browser.contexts()[0] || await browser.newContext();
  let page = context.pages().find((candidate: any) => candidate.url().startsWith('https://suno.com/create'));
  if (!page) page = await context.newPage();

  await page.goto('https://suno.com/create', { waitUntil: 'domcontentloaded', timeout: 30000 }).catch(() => undefined);
  await page.waitForLoadState('domcontentloaded', { timeout: 10000 }).catch(() => undefined);
  await page.waitForTimeout(2000).catch(() => undefined);
  await page.bringToFront().catch(() => undefined);
  await page.locator('#codex-suno-manual-panel button[data-close="1"]').click({ timeout: 1000 }).catch(() => undefined);
  await closeSunoDialog(page);

  await page.getByRole('tab', { name: /^Advanced$/i }).click({ timeout: 5000 }).catch(() => undefined);
  await selectSunoWorkspace(page, body.sunoProjectName);

  if (body.instrumental) {
    await page.getByRole('radio', { name: /^Instrumental$/i }).click({ timeout: 3000 }).catch(() => undefined);
  } else {
    await page.getByRole('radio', { name: /^Write$/i }).click({ timeout: 3000 }).catch(() => undefined);
    const lyricsArea = page.locator('textarea:visible').nth(0);
    await lyricsArea.fill(body.lyrics || '', { timeout: 5000 });
  }

  const styleArea = page.locator('textarea:visible').nth(1);
  await styleArea.fill(body.style || '', { timeout: 5000 });
  const titleInputs = page.locator('input[placeholder="Song Title (Optional)"]:visible');
  const titleInputCount = await titleInputs.count().catch(() => 0);
  for (let index = 0; index < titleInputCount; index += 1) {
    await titleInputs.nth(index).fill(body.title || '', { timeout: 5000 }).catch(() => undefined);
  }

  const moreOptions = page.getByText(/^More Options$/i).first();
  if (await moreOptions.isVisible({ timeout: 3000 }).catch(() => false)) {
    await moreOptions.click({ timeout: 5000 }).catch(() => undefined);
    await page.waitForTimeout(500);
  }

  if (body.sunoVocalGender === 'm') {
    await page.getByRole('button', { name: /^Male$/i }).click({ timeout: 3000 }).catch(() => undefined);
  } else if (body.sunoVocalGender === 'f') {
    await page.getByRole('button', { name: /^Female$/i }).click({ timeout: 3000 }).catch(() => undefined);
  }
  await setSunoSlider(page, 'Weirdness', body.sunoWeirdness);
  await setSunoSlider(page, 'Style Influence', body.sunoStyleInfluence);
  await closeSunoDialog(page);

  const shouldSubmit = body.sunoBrowserSubmit !== false;
  if (shouldSubmit) {
    const createButton = page.getByRole('button', { name: /Create song|^Create$/i }).last();
    await createButton.click({ timeout: 10000 });
  }

  return {
    browserAutomated: true,
    submitted: shouldSubmit,
    workspaceName: body.sunoProjectName || null,
    cdpUrl: SUNO_CHROME_CDP_URL,
    pageUrl: await page.url(),
    message: shouldSubmit
      ? 'Suno 화면에 설정값을 입력하고 Create를 눌렀습니다. CAPTCHA가 표시되면 Cloakbrowser에서 직접 완료하세요.'
      : 'Suno 화면에 설정값을 입력하고 workspace를 선택했습니다. 검증 모드라 Create는 누르지 않았습니다.',
  };
}

async function parseJsonOrText(response: globalThis.Response): Promise<unknown> {
  const rawText = await response.text();
  try {
    return rawText ? JSON.parse(rawText) : {};
  } catch {
    return rawText;
  }
}

function collectAudioUrls(value: unknown, urls = new Set<string>()): Set<string> {
  if (!value) return urls;
  if (typeof value === 'string') {
    if (/^https?:\/\/.+\.(mp3|wav|flac|m4a|aac|ogg|opus)(\?.*)?$/i.test(value)) {
      urls.add(value);
    }
    return urls;
  }
  if (Array.isArray(value)) {
    value.forEach(item => collectAudioUrls(item, urls));
    return urls;
  }
  if (typeof value === 'object') {
    for (const [key, nested] of Object.entries(value as Record<string, unknown>)) {
      const lowerKey = key.toLowerCase();
      if (
        typeof nested === 'string' &&
        (lowerKey.includes('audio') || lowerKey === 'url' || lowerKey.includes('stream'))
      ) {
        collectAudioUrls(nested, urls);
      } else {
        collectAudioUrls(nested, urls);
      }
    }
  }
  return urls;
}

function collectClipIds(value: unknown, ids = new Set<string>()): Set<string> {
  if (!value) return ids;
  if (typeof value === 'string') return ids;
  if (Array.isArray(value)) {
    value.forEach(item => collectClipIds(item, ids));
    return ids;
  }
  if (typeof value === 'object') {
    const obj = value as Record<string, unknown>;
    if (typeof obj.id === 'string' && ('status' in obj || 'metadata' in obj || 'audio_url' in obj)) {
      ids.add(obj.id);
    }
    for (const nested of Object.values(obj)) {
      collectClipIds(nested, ids);
    }
  }
  return ids;
}

async function pollSunoClips(baseUrl: string, bearerToken: string | undefined, clipIds: string[], seconds: number): Promise<unknown> {
  const deadline = Date.now() + Math.max(1, seconds) * 1000;
  let lastResponse: unknown = {};
  while (Date.now() < deadline) {
    const url = buildSunoUrl(baseUrl, `/api/feed/?ids=${encodeURIComponent(clipIds.join(','))}`);
    const response = await fetch(url, { headers: await buildSunoHeaders(bearerToken) });
    lastResponse = await parseJsonOrText(response);
    const audioUrls = Array.from(collectAudioUrls(lastResponse));
    if (audioUrls.length > 0) return lastResponse;
    await new Promise(resolve => setTimeout(resolve, 3000));
  }
  return lastResponse;
}

function inferAudioExtension(url: string): string {
  const clean = url.split('?')[0].toLowerCase();
  if (clean.endsWith('.flac')) return '.flac';
  if (clean.endsWith('.wav')) return '.wav';
  if (clean.endsWith('.m4a')) return '.m4a';
  if (clean.endsWith('.aac')) return '.aac';
  if (clean.endsWith('.ogg')) return '.ogg';
  if (clean.endsWith('.opus')) return '.opus';
  return '.mp3';
}

router.post('/upload-audio', authMiddleware, (req: AuthenticatedRequest, res: Response, next: Function) => {
  audioUpload.single('audio')(req, res, (err: any) => {
    if (err) {
      res.status(400).json({ error: err.message || 'Invalid file upload' });
      return;
    }
    next();
  });
}, async (req: AuthenticatedRequest, res: Response) => {
  try {
    if (!req.file) {
      res.status(400).json({ error: 'Audio file is required' });
      return;
    }

    const storage = getStorageProvider();
    const extFromName = path.extname(req.file.originalname || '').toLowerCase();
    const extFromType = (() => {
      switch (req.file.mimetype) {
        case 'audio/mpeg':
          return '.mp3';
        case 'audio/wav':
        case 'audio/x-wav':
          return '.wav';
        case 'audio/flac':
        case 'audio/x-flac':
          return '.flac';
        case 'audio/ogg':
          return '.ogg';
        case 'audio/mp4':
        case 'audio/x-m4a':
        case 'audio/aac':
          return '.m4a';
        case 'audio/webm':
          return '.webm';
        case 'video/mp4':
          return '.mp4';
        default:
          return '';
      }
    })();
    const ext = extFromName || extFromType || '.audio';
    const key = `references/${req.user!.id}/${Date.now()}-${generateUUID()}${ext}`;
    const storedKey = await storage.upload(key, req.file.buffer, req.file.mimetype);
    const publicUrl = storage.getPublicUrl(storedKey);

    res.json({ url: publicUrl, key: storedKey });
  } catch (error) {
    console.error('Upload reference audio error:', error);
    res.status(500).json({ error: 'Failed to upload audio' });
  }
});

router.post('/suno/projects/list', authMiddleware, async (req: AuthenticatedRequest, res: Response) => {
  try {
    const { baseUrl = 'https://studio-api.prod.suno.com', apiKey } = req.body as { baseUrl?: string; apiKey?: string };
    const bearerToken = await getSunoBearerToken(apiKey);
    const response = await fetch(buildSunoUrl(baseUrl, '/api/project/me'), {
      headers: await buildSunoHeaders(bearerToken),
    });
    const raw = await parseJsonOrText(response);
    if (!response.ok) {
      res.status(response.status).json({ error: 'Failed to load Suno projects', raw });
      return;
    }
    const projects = Array.isArray(raw)
      ? raw
      : Array.isArray((raw as any)?.projects)
        ? (raw as any).projects
        : Array.isArray((raw as any)?.data)
          ? (raw as any).data
          : [];
    res.json({ projects, raw });
  } catch (error) {
    console.error('Suno project list error:', error);
    res.status(500).json({ error: (error as Error).message || 'Failed to load Suno projects' });
  }
});

router.post('/suno/projects', authMiddleware, async (req: AuthenticatedRequest, res: Response) => {
  try {
    const { baseUrl = 'https://studio-api.prod.suno.com', apiKey, name } = req.body as { baseUrl?: string; apiKey?: string; name?: string };
    if (!name?.trim()) {
      res.status(400).json({ error: 'Project name is required' });
      return;
    }
    const bearerToken = await getSunoBearerToken(apiKey);
    const response = await fetch(buildSunoUrl(baseUrl, '/api/project'), {
      method: 'POST',
      headers: await buildSunoHeaders(bearerToken),
      body: JSON.stringify({ name: name.trim(), description: '' }),
    });
    const raw = await parseJsonOrText(response);
    if (!response.ok) {
      res.status(response.status).json({ error: 'Failed to create Suno project', raw });
      return;
    }
    const project = (raw as any)?.project || (raw as any)?.data || raw;
    res.json({ project, raw });
  } catch (error) {
    console.error('Suno project create error:', error);
    res.status(500).json({ error: (error as Error).message || 'Failed to create Suno project' });
  }
});

router.post('/suno/projects/clips', authMiddleware, async (req: AuthenticatedRequest, res: Response) => {
  try {
    const {
      baseUrl = 'https://studio-api.prod.suno.com',
      apiKey,
      projectId,
      page = 1,
      query,
    } = req.body as { baseUrl?: string; apiKey?: string; projectId?: string; page?: number; query?: string };
    if (!projectId?.trim()) {
      res.status(400).json({ error: 'projectId is required' });
      return;
    }

    const bearerToken = await getSunoBearerToken(apiKey);
    const url = new URL(buildSunoUrl(baseUrl, `/api/project/${encodeURIComponent(projectId.trim())}`));
    url.searchParams.set('page', String(page || 1));
    if (query?.trim()) url.searchParams.set('query', query.trim());

    const response = await fetch(url.toString(), {
      headers: await buildSunoHeaders(bearerToken),
    });
    const raw = await parseJsonOrText(response);
    if (!response.ok) {
      res.status(response.status).json({ error: 'Failed to load Suno workspace clips', raw });
      return;
    }

    const projectClips = Array.isArray((raw as any)?.project_clips) ? (raw as any).project_clips : [];
    const pinnedClips = Array.isArray((raw as any)?.pinned_clips) ? (raw as any).pinned_clips : [];
    const clipsById = new Map<string, unknown>();
    [...pinnedClips, ...projectClips].forEach((item: any) => {
      const clip = item?.clip || item;
      if (clip?.id) clipsById.set(clip.id, clip);
    });

    res.json({
      project: {
        id: (raw as any)?.id || projectId,
        name: (raw as any)?.name,
        description: (raw as any)?.description,
        clip_count: (raw as any)?.clip_count,
      },
      clips: Array.from(clipsById.values()),
      raw,
    });
  } catch (error) {
    console.error('Suno project clips error:', error);
    res.status(500).json({ error: (error as Error).message || 'Failed to load Suno workspace clips' });
  }
});

router.post('/suno/projects/add-clips', authMiddleware, async (req: AuthenticatedRequest, res: Response) => {
  try {
    const { baseUrl = 'https://studio-api.prod.suno.com', apiKey, projectId, clipIds } = req.body as { baseUrl?: string; apiKey?: string; projectId?: string; clipIds?: string[] };
    if (!projectId || !Array.isArray(clipIds) || clipIds.length === 0) {
      res.status(400).json({ error: 'projectId and clipIds are required' });
      return;
    }
    const bearerToken = await getSunoBearerToken(apiKey);
    const response = await fetch(buildSunoUrl(baseUrl, `/api/project/${encodeURIComponent(projectId)}/clips`), {
      method: 'POST',
      headers: await buildSunoHeaders(bearerToken),
      body: JSON.stringify({
        update_type: 'add',
        metadata: { clip_ids: clipIds },
      }),
    });
    const raw = await parseJsonOrText(response);
    if (!response.ok) {
      res.status(response.status).json({ error: 'Failed to add Suno clips to project', raw });
      return;
    }
    res.json({ raw });
  } catch (error) {
    console.error('Suno add clips error:', error);
    res.status(500).json({ error: (error as Error).message || 'Failed to add Suno clips to project' });
  }
});

router.post('/suno/projects/remove-clips', authMiddleware, async (req: AuthenticatedRequest, res: Response) => {
  try {
    const { baseUrl = 'https://studio-api.prod.suno.com', apiKey, projectId, clipIds } = req.body as { baseUrl?: string; apiKey?: string; projectId?: string; clipIds?: string[] };
    if (!projectId || !Array.isArray(clipIds) || clipIds.length === 0) {
      res.status(400).json({ error: 'projectId and clipIds are required' });
      return;
    }
    const bearerToken = await getSunoBearerToken(apiKey);
    const response = await fetch(buildSunoUrl(baseUrl, `/api/project/${encodeURIComponent(projectId)}/clips`), {
      method: 'POST',
      headers: await buildSunoHeaders(bearerToken),
      body: JSON.stringify({
        update_type: 'remove',
        metadata: { clip_ids: clipIds },
      }),
    });
    const raw = await parseJsonOrText(response);
    if (!response.ok) {
      res.status(response.status).json({ error: 'Failed to remove Suno clips from project', raw });
      return;
    }
    res.json({ raw });
  } catch (error) {
    console.error('Suno remove clips error:', error);
    res.status(500).json({ error: (error as Error).message || 'Failed to remove Suno clips from project' });
  }
});

router.post('/suno/upsample-tags', authMiddleware, async (req: AuthenticatedRequest, res: Response) => {
  try {
    const {
      baseUrl = 'https://studio-api.prod.suno.com',
      apiKey,
      tags,
    } = req.body as { baseUrl?: string; apiKey?: string; tags?: string };
    if (!tags?.trim()) {
      res.status(400).json({ error: 'tags is required' });
      return;
    }

    const bearerToken = await getSunoBearerToken(apiKey);
    const response = await fetch(buildSunoUrl(baseUrl, '/api/generate/upsample-tags'), {
      method: 'POST',
      headers: await buildSunoHeaders(bearerToken),
      body: JSON.stringify({ original_tags: tags.trim() }),
    });
    const raw = await parseJsonOrText(response);
    if (!response.ok) {
      res.status(response.status).json({ error: 'Failed to personalize Suno style prompt', raw });
      return;
    }

    const style = typeof raw === 'string'
      ? raw
      : ((raw as any)?.upsampled_tags || (raw as any)?.style || (raw as any)?.text || '');
    res.json({ style, raw });
  } catch (error) {
    console.error('Suno upsample tags error:', error);
    res.status(500).json({ error: (error as Error).message || 'Failed to personalize Suno style prompt' });
  }
});

router.post('/suno', authMiddleware, async (req: AuthenticatedRequest, res: Response) => {
  try {
    const body = req.body as GenerateBody;
    const baseUrl = body.sunoBaseUrl?.trim() || 'https://studio-api.prod.suno.com';
    if (!baseUrl) {
      res.status(400).json({ error: 'Suno Base URL is required' });
      return;
    }

    const endpoint = body.sunoEndpoint?.trim() || '/api/generate/v2-web/';
    const targetUrl = buildSunoUrl(baseUrl, endpoint);
    const prompt = body.style || body.songDescription || body.title || 'music';
    const createMode = body.customMode === false ? 'SIMPLE' : 'CUSTOM';
    const metadata: Record<string, unknown> = {
      create_mode: createMode,
      ...(createMode === 'SIMPLE' ? { lyrics_model: 'default' } : {}),
    };
    if (body.sunoVocalGender === 'm' || body.sunoVocalGender === 'f') {
      metadata.vocal_gender = body.sunoVocalGender;
    }

    const controlSliders: Record<string, number> = {};
    const weirdness = clampSunoSlider(body.sunoWeirdness);
    if (weirdness !== undefined && weirdness !== 50) {
      controlSliders.weirdness_constraint = weirdness / 100;
    }
    const styleInfluence = clampSunoSlider(body.sunoStyleInfluence);
    if (styleInfluence !== undefined && styleInfluence !== 50) {
      controlSliders.style_weight = styleInfluence / 100;
    }
    if (Object.keys(controlSliders).length > 0) {
      metadata.control_sliders = controlSliders;
    }

    const payload: Record<string, unknown> = {
      prompt: createMode === 'CUSTOM' ? (body.instrumental ? '' : (body.lyrics || '')) : '',
      gpt_description_prompt: createMode === 'SIMPLE' ? (body.songDescription || prompt) : '',
      tags: body.style,
      negative_tags: body.sunoNegativeTags?.trim() || '',
      title: body.title,
      mv: body.sunoModel || 'chirp-fenix',
      make_instrumental: Boolean(body.instrumental),
      generation_type: 'TEXT',
      metadata,
      project_id: body.sunoProjectId?.trim() || null,
      token: null,
    };
    addOptionalString(payload, 'task', body.sunoTask);
    addOptionalString(payload, 'continue_clip_id', body.sunoContinueClipId);
    addOptionalNumber(payload, 'continue_at', body.sunoContinueAt);
    addOptionalString(payload, 'cover_clip_id', body.sunoCoverClipId);
    addOptionalString(payload, 'artist_clip_id', body.sunoArtistClipId);
    addOptionalNumber(payload, 'infill_start_s', body.sunoInfillStartS);
    addOptionalNumber(payload, 'infill_end_s', body.sunoInfillEndS);
    addOptionalString(payload, 'stem_type_id', body.sunoStemTypeId);
    const overrideFields = parseOptionalJsonObject(body.sunoOverrideFieldsJson);
    if (overrideFields) payload.override_fields = overrideFields;

    const bearerToken = await getSunoBearerToken(body.sunoApiKey);
    const headers = await buildSunoHeaders(bearerToken);

    const sunoResponse = await fetch(targetUrl, {
      method: 'POST',
      headers,
      body: JSON.stringify(payload),
    });

    let external = await parseJsonOrText(sunoResponse);

    if (!sunoResponse.ok) {
      const message = typeof external === 'object' && external !== null
        ? ((external as any).error || (external as any).message || JSON.stringify(external))
        : String(external);
      if (isManualSunoFallbackError(sunoResponse.status, external)) {
        const manual = await runSunoBrowserGenerate(body);
        res.status(202).json({ songs: [], external: { ...manual, apiError: external } });
        return;
      }
      res.status(sunoResponse.status).json({ error: message || 'Suno API request failed', external });
      return;
    }

    let audioUrls = Array.from(collectAudioUrls(external));
    const clipIds = Array.from(collectClipIds(external));
    if (audioUrls.length === 0 && body.sunoWaitForAudio !== false && clipIds.length > 0) {
      external = await pollSunoClips(baseUrl, bearerToken, clipIds, body.sunoPollSeconds || 90);
      audioUrls = Array.from(collectAudioUrls(external));
    }

    if (audioUrls.length === 0) {
      res.json({ songs: [], clips: clipIds, external });
      return;
    }

    const storage = getStorageProvider();
    const songs = [];

    for (let i = 0; i < audioUrls.length; i++) {
      const audioUrl = audioUrls[i];
      const songId = generateUUID();
      const variationSuffix = audioUrls.length > 1 ? ` (v${i + 1})` : '';
      const songTitle = autoTitle(body) + variationSuffix;
      let storedPath = audioUrl;

      try {
        const { buffer } = await downloadAudioToBuffer(audioUrl);
        const ext = inferAudioExtension(audioUrl);
        const storageKey = `${req.user!.id}/${songId}${ext}`;
        await storage.upload(storageKey, buffer, `audio/${ext.slice(1)}`);
        storedPath = storage.getPublicUrl(storageKey);
      } catch (downloadError) {
        console.warn(`Suno audio download failed, keeping remote URL: ${audioUrl}`, downloadError);
      }

      await pool.query(
        `INSERT INTO songs (id, user_id, title, lyrics, style, caption, audio_url,
                            duration, bpm, key_scale, time_signature, tags, is_public, generation_params,
                            created_at, updated_at)
         VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 1, ?, datetime('now'), datetime('now'))`,
        [
          songId,
          req.user!.id,
          songTitle,
          body.instrumental ? '[Instrumental]' : (body.lyrics || ''),
          body.style || '',
          body.style || prompt,
          storedPath,
          body.duration && body.duration > 0 ? body.duration : 0,
          body.bpm || 0,
          body.keyScale || '',
          body.timeSignature || '',
          JSON.stringify(['suno']),
          JSON.stringify({ ...body, sunoApiKey: body.sunoApiKey ? '[redacted]' : undefined, sunoResponse: external }),
        ]
      );

      songs.push({
        id: songId,
        title: songTitle,
        lyrics: body.instrumental ? '[Instrumental]' : (body.lyrics || ''),
        style: body.style || '',
        audio_url: storedPath,
        duration: body.duration && body.duration > 0 ? body.duration : 0,
        tags: ['suno'],
        is_public: true,
        created_at: new Date().toISOString(),
      });
    }

    res.json({ songs, external });
  } catch (error) {
    console.error('Suno generate error:', error);
    res.status(500).json({ error: (error as Error).message || 'Suno generation failed' });
  }
});

router.post('/', authMiddleware, async (req: AuthenticatedRequest, res: Response) => {
  try {
    const {
      customMode,
      songDescription,
      lyrics,
      style,
      title,
      instrumental,
      vocalLanguage,
      duration,
      bpm,
      keyScale,
      timeSignature,
      inferenceSteps,
      guidanceScale,
      batchSize,
      randomSeed,
      seed,
      thinking,
      audioFormat,
      inferMethod,
      shift,
      lmTemperature,
      lmCfgScale,
      lmTopK,
      lmTopP,
      lmNegativePrompt,
      lmBackend,
      lmModel,
      referenceAudioUrl,
      sourceAudioUrl,
      referenceAudioTitle,
      sourceAudioTitle,
      audioCodes,
      repaintingStart,
      repaintingEnd,
      instruction,
      audioCoverStrength,
      taskType,
      useAdg,
      cfgIntervalStart,
      cfgIntervalEnd,
      customTimesteps,
      useCotMetas,
      useCotCaption,
      useCotLanguage,
      autogen,
      constrainedDecodingDebug,
      allowLmBatch,
      getScores,
      getLrc,
      scoreScale,
      lmBatchChunkSize,
      trackName,
      completeTrackClasses,
      isFormatCaption,
      ditModel,
    } = req.body as GenerateBody;

    if (!customMode && !songDescription) {
      res.status(400).json({ error: 'Song description required for simple mode' });
      return;
    }

    if (customMode && !style && !lyrics && !referenceAudioUrl) {
      res.status(400).json({ error: 'Style, lyrics, or reference audio required for custom mode' });
      return;
    }

    const params = {
      customMode,
      songDescription,
      lyrics,
      style,
      title,
      instrumental,
      vocalLanguage,
      duration,
      bpm,
      keyScale,
      timeSignature,
      inferenceSteps,
      guidanceScale,
      batchSize,
      randomSeed,
      seed,
      thinking,
      audioFormat,
      inferMethod,
      shift,
      lmTemperature,
      lmCfgScale,
      lmTopK,
      lmTopP,
      lmNegativePrompt,
      lmBackend,
      lmModel,
      referenceAudioUrl,
      sourceAudioUrl,
      referenceAudioTitle,
      sourceAudioTitle,
      audioCodes,
      repaintingStart,
      repaintingEnd,
      instruction,
      audioCoverStrength,
      taskType,
      useAdg,
      cfgIntervalStart,
      cfgIntervalEnd,
      customTimesteps,
      useCotMetas,
      useCotCaption,
      useCotLanguage,
      autogen,
      constrainedDecodingDebug,
      allowLmBatch,
      getScores,
      getLrc,
      scoreScale,
      lmBatchChunkSize,
      trackName,
      completeTrackClasses,
      isFormatCaption,
      ditModel,
    };

    // Create job record in database
    const localJobId = generateUUID();
    await pool.query(
      `INSERT INTO generation_jobs (id, user_id, status, params, created_at, updated_at)
       VALUES (?, ?, 'queued', ?, datetime('now'), datetime('now'))`,
      [localJobId, req.user!.id, JSON.stringify(params)]
    );

    // Start generation
    const { jobId: hfJobId } = await generateMusicViaAPI(params);

    // Update job with ACE-Step task ID
    await pool.query(
      `UPDATE generation_jobs SET acestep_task_id = ?, status = 'running', updated_at = datetime('now') WHERE id = ?`,
      [hfJobId, localJobId]
    );

    res.json({
      jobId: localJobId,
      status: 'queued',
      queuePosition: 1,
    });
  } catch (error) {
    console.error('Generate error:', error);
    res.status(500).json({ error: (error as Error).message || 'Generation failed' });
  }
});

router.get('/status/:jobId', authMiddleware, async (req: AuthenticatedRequest, res: Response) => {
  try {
    const jobResult = await pool.query(
      `SELECT id, user_id, acestep_task_id, status, params, result, error, created_at
       FROM generation_jobs
       WHERE id = ?`,
      [req.params.jobId]
    );

    if (jobResult.rows.length === 0) {
      res.status(404).json({ error: 'Job not found' });
      return;
    }

    const job = jobResult.rows[0];

    if (job.user_id !== req.user!.id) {
      res.status(403).json({ error: 'Access denied' });
      return;
    }

    // If job is still running, check ACE-Step status
    if (['pending', 'queued', 'running'].includes(job.status) && job.acestep_task_id) {
      try {
        const aceStatus = await getJobStatus(job.acestep_task_id);

        if (aceStatus.status !== job.status) {
          // Use optimistic lock: only update if status hasn't changed (prevents duplicate song creation)
          let updateQuery = `UPDATE generation_jobs SET status = ?, updated_at = datetime('now')`;
          const updateParams: unknown[] = [aceStatus.status];

          if (aceStatus.status === 'succeeded' && aceStatus.result) {
            updateQuery += `, result = ?`;
            updateParams.push(JSON.stringify(aceStatus.result));
          } else if (aceStatus.status === 'failed' && aceStatus.error) {
            updateQuery += `, error = ?`;
            updateParams.push(aceStatus.error);
          }

          updateQuery += ` WHERE id = ? AND status = ?`;
          updateParams.push(req.params.jobId, job.status);

          const updateResult = await pool.query(updateQuery, updateParams);
          const wasUpdated = updateResult.rowCount > 0;

          // If succeeded AND we were the first to update (optimistic lock), create song records
          if (aceStatus.status === 'succeeded' && aceStatus.result && wasUpdated) {
            const params = typeof job.params === 'string' ? JSON.parse(job.params) : job.params;
            const audioUrls = aceStatus.result.audioUrls.filter((url: string) => {
              const lower = url.toLowerCase();
              return lower.endsWith('.mp3') || lower.endsWith('.flac') || lower.endsWith('.wav');
            });
            const localPaths: string[] = [];
            const storage = getStorageProvider();

            for (let i = 0; i < audioUrls.length; i++) {
              const audioUrl = audioUrls[i];
              const variationSuffix = audioUrls.length > 1 ? ` (v${i + 1})` : '';
              const songTitle = autoTitle(params) + variationSuffix;

              const songId = generateUUID();

              try {
                const { buffer } = await downloadAudioToBuffer(audioUrl);
                const ext = audioUrl.includes('.flac') ? '.flac' : '.mp3';
                const storageKey = `${req.user!.id}/${songId}${ext}`;
                await storage.upload(storageKey, buffer, `audio/${ext.slice(1)}`);
                const storedPath = storage.getPublicUrl(storageKey);

                await pool.query(
                  `INSERT INTO songs (id, user_id, title, lyrics, style, caption, audio_url,
                                      duration, bpm, key_scale, time_signature, tags, is_public, generation_params,
                                      created_at, updated_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 1, ?, datetime('now'), datetime('now'))`,
                  [
                    songId,
                    req.user!.id,
                    songTitle,
                    params.instrumental ? '[Instrumental]' : params.lyrics,
                    params.style,
                    params.style,
                    storedPath,
                    aceStatus.result.duration && aceStatus.result.duration > 0 ? aceStatus.result.duration : (params.duration && params.duration > 0 ? params.duration : 0),
                    aceStatus.result.bpm || params.bpm,
                    aceStatus.result.keyScale || params.keyScale,
                    aceStatus.result.timeSignature || params.timeSignature,
                    JSON.stringify([]),
                    JSON.stringify(params),
                  ]
                );

                localPaths.push(storedPath);
              } catch (downloadError) {
                console.error(`Failed to download audio ${i + 1}:`, downloadError);
                // Still create song record with remote URL
                await pool.query(
                  `INSERT INTO songs (id, user_id, title, lyrics, style, caption, audio_url,
                                      duration, bpm, key_scale, time_signature, tags, is_public, generation_params,
                                      created_at, updated_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 1, ?, datetime('now'), datetime('now'))`,
                  [
                    songId,
                    req.user!.id,
                    songTitle,
                    params.instrumental ? '[Instrumental]' : params.lyrics,
                    params.style,
                    params.style,
                    audioUrl,
                    aceStatus.result.duration && aceStatus.result.duration > 0 ? aceStatus.result.duration : (params.duration && params.duration > 0 ? params.duration : 0),
                    aceStatus.result.bpm || params.bpm,
                    aceStatus.result.keyScale || params.keyScale,
                    aceStatus.result.timeSignature || params.timeSignature,
                    JSON.stringify([]),
                    JSON.stringify(params),
                  ]
                );
                localPaths.push(audioUrl);
              }
            }

            aceStatus.result.audioUrls = localPaths;
            cleanupJob(job.acestep_task_id);
          }
        }

        res.json({
          jobId: req.params.jobId,
          status: aceStatus.status,
          queuePosition: aceStatus.queuePosition,
          etaSeconds: aceStatus.etaSeconds,
          progress: aceStatus.progress,
          stage: aceStatus.stage,
          result: aceStatus.result,
          error: aceStatus.error,
        });
        return;
      } catch (aceError) {
        console.error('ACE-Step status check error:', aceError);
      }
    }

    // Return stored status
    res.json({
      jobId: req.params.jobId,
      status: job.status,
      progress: undefined,
      stage: undefined,
      result: job.result && typeof job.result === 'string' ? JSON.parse(job.result) : job.result,
      error: job.error,
    });
  } catch (error) {
    console.error('Status check error:', error);
    res.status(500).json({ error: 'Internal server error' });
  }
});

// Audio proxy endpoint
router.get('/audio', async (req, res: Response) => {
  try {
    const audioPath = req.query.path as string;
    if (!audioPath) {
      res.status(400).json({ error: 'Path required' });
      return;
    }

    const audioResponse = await getAudioStream(audioPath);

    if (!audioResponse.ok) {
      res.status(audioResponse.status).json({ error: 'Failed to fetch audio' });
      return;
    }

    const contentType = audioResponse.headers.get('content-type');
    if (contentType) {
      res.setHeader('Content-Type', contentType);
    }

    const contentLength = audioResponse.headers.get('content-length');
    if (contentLength) {
      res.setHeader('Content-Length', contentLength);
    }

    const reader = audioResponse.body?.getReader();
    if (!reader) {
      res.status(500).json({ error: 'Failed to read audio stream' });
      return;
    }

    const pump = async (): Promise<void> => {
      const { done, value } = await reader.read();
      if (done) {
        res.end();
        return;
      }
      res.write(value);
      return pump();
    };

    await pump();
  } catch (error) {
    console.error('Audio proxy error:', error);
    res.status(500).json({ error: 'Internal server error' });
  }
});

router.get('/history', authMiddleware, async (req: AuthenticatedRequest, res: Response) => {
  try {
    const result = await pool.query(
      `SELECT id, acestep_task_id, status, params, result, error, created_at
       FROM generation_jobs
       WHERE user_id = ?
       ORDER BY created_at DESC
       LIMIT 50`,
      [req.user!.id]
    );

    res.json({ jobs: result.rows });
  } catch (error) {
    console.error('Get history error:', error);
    res.status(500).json({ error: 'Internal server error' });
  }
});

router.get('/endpoints', authMiddleware, async (_req: AuthenticatedRequest, res: Response) => {
  try {
    const endpoints = await discoverEndpoints();
    res.json({ endpoints });
  } catch (error) {
    console.error('Discover endpoints error:', error);
    res.status(500).json({ error: 'Failed to discover endpoints' });
  }
});

router.get('/models', async (_req, res: Response) => {
  try {
    const ACESTEP_DIR = process.env.ACESTEP_PATH || path.resolve(path.dirname(fileURLToPath(import.meta.url)), '../../../ACE-Step-1.5');
    const checkpointsDir = path.join(ACESTEP_DIR, 'checkpoints');

    // All known DiT models from Gradio's model_downloader.py registry:
    // - MAIN_MODEL_COMPONENTS includes "acestep-v15-turbo" (bundled with main download)
    // - SUBMODEL_REGISTRY includes the rest (separate HuggingFace repos, auto-downloaded on init)
    const ALL_DIT_MODELS = [
      'acestep-v15-turbo',             // default, from main model repo
      'acestep-v15-base',              // submodel
      'acestep-v15-sft',               // submodel
      'acestep-v15-turbo-shift1',      // submodel
      'acestep-v15-turbo-shift3',      // submodel
      'acestep-v15-turbo-continuous',  // submodel
      // 4B XL models (≈9GB download, need GPU offload — best on T4+ GPU, not CPU):
      'acestep-v15-xl-turbo',          // 4B XL turbo — best practical quality (recommended for T4)
      'acestep-v15-xl-base',           // 4B XL — high quality, 50 steps + CFG
      'acestep-v15-xl-sft',            // 4B XL — highest quality, 50 steps + CFG
    ];

    // Query Gradio /v1/models to get the currently loaded/active model
    let activeModel: string | null = null;
    try {
      const apiRes = await fetch(`${config.acestep.apiUrl}/v1/models`);
      if (apiRes.ok) {
        const data = await apiRes.json() as any;
        const gradioModels = data?.data?.models || data?.models || [];
        if (gradioModels.length > 0) {
          activeModel = gradioModels[0]?.name || null;
        }
      }
    } catch {
      // Gradio API unavailable
    }

    // Check which models are downloaded (exist on disk)
    // Matches Gradio's handler.py check_model_exists() and get_available_acestep_v15_models()
    const { existsSync, statSync } = await import('fs');
    const downloaded = new Set<string>();
    for (const model of ALL_DIT_MODELS) {
      const modelPath = path.join(checkpointsDir, model);
      try {
        if (existsSync(modelPath) && statSync(modelPath).isDirectory()) {
          downloaded.add(model);
        }
      } catch { /* skip */ }
    }

    // Also scan for any additional acestep-v15-* models on disk not in the registry
    // (e.g. user-trained or community models)
    try {
      const { readdirSync } = await import('fs');
      for (const entry of readdirSync(checkpointsDir)) {
        if (entry.startsWith('acestep-v15-') && statSync(path.join(checkpointsDir, entry)).isDirectory()) {
          downloaded.add(entry);
          if (!ALL_DIT_MODELS.includes(entry)) {
            ALL_DIT_MODELS.push(entry);
          }
        }
      }
    } catch { /* checkpoints dir may not exist */ }

    const models = ALL_DIT_MODELS.map(name => ({
      name,
      is_active: name === activeModel,
      is_preloaded: downloaded.has(name),
    }));

    // Sort: active first, then downloaded, then alphabetical
    models.sort((a, b) => {
      if (a.is_active !== b.is_active) return a.is_active ? -1 : 1;
      if (a.is_preloaded !== b.is_preloaded) return a.is_preloaded ? -1 : 1;
      return a.name.localeCompare(b.name);
    });

    res.json({ models });
  } catch (error) {
    console.error('Models error:', error);
    res.status(500).json({ error: (error as Error).message });
  }
});

// GET /api/generate/random-description — Load a random simple description from Gradio
router.get('/random-description', authMiddleware, async (_req: AuthenticatedRequest, res: Response) => {
  try {
    const client = await getGradioClient();
    const result = await client.predict('/load_random_simple_description', []);
    const data = result.data as unknown[];
    // Returns [description, instrumental, vocal_language]
    res.json({
      description: data[0] || '',
      instrumental: data[1] || false,
      vocalLanguage: data[2] || 'unknown',
    });
  } catch (error) {
    console.error('Random description error:', error);
    res.status(500).json({ error: (error as Error).message });
  }
});

router.get('/health', async (_req, res: Response) => {
  try {
    const healthy = await checkSpaceHealth();
    res.json({ healthy, aceStepUrl: config.acestep.apiUrl });
  } catch (error) {
    res.json({ healthy: false, aceStepUrl: config.acestep.apiUrl, error: (error as Error).message });
  }
});

router.get('/limits', async (_req, res: Response) => {
  try {
    const { spawn } = await import('child_process');
    const ACESTEP_DIR = process.env.ACESTEP_PATH || path.resolve(path.dirname(fileURLToPath(import.meta.url)), '../../../ACE-Step-1.5');
    const __filename = fileURLToPath(import.meta.url);
    const __dirname = path.dirname(__filename);
    const SCRIPTS_DIR = path.join(__dirname, '../../scripts');
    const LIMITS_SCRIPT = path.join(SCRIPTS_DIR, 'get_limits.py');
    const pythonPath = resolvePythonPath(ACESTEP_DIR);

    const result = await new Promise<{ success: boolean; data?: any; error?: string }>((resolve) => {
      const proc = spawn(pythonPath, [LIMITS_SCRIPT], {
        cwd: ACESTEP_DIR,
        env: {
          ...process.env,
          ACESTEP_PATH: ACESTEP_DIR,
        },
      });

      let stdout = '';
      let stderr = '';

      proc.stdout.on('data', (data) => { stdout += data.toString(); });
      proc.stderr.on('data', (data) => { stderr += data.toString(); });

      proc.on('close', (code) => {
        if (code === 0 && stdout) {
          try {
            const parsed = JSON.parse(stdout);
            resolve({ success: true, data: parsed });
          } catch {
            resolve({ success: false, error: 'Failed to parse limits result' });
          }
        } else {
          resolve({ success: false, error: stderr || 'Failed to read limits' });
        }
      });

      proc.on('error', (err) => {
        resolve({ success: false, error: err.message });
      });
    });

    if (result.success && result.data) {
      res.json(result.data);
    } else {
      res.status(500).json({ error: result.error || 'Failed to load limits' });
    }
  } catch (error) {
    console.error('Limits error:', error);
    res.status(500).json({ error: (error as Error).message });
  }
});

router.get('/debug/:taskId', authMiddleware, async (req: AuthenticatedRequest, res: Response) => {
  try {
    const rawResponse = getJobRawResponse(req.params.taskId);
    if (!rawResponse) {
      res.status(404).json({ error: 'Job not found or no raw response available' });
      return;
    }
    res.json({ rawResponse });
  } catch (error) {
    res.status(500).json({ error: (error as Error).message });
  }
});

// Format endpoint - uses LLM to enhance style/lyrics
router.post('/format', authMiddleware, async (req: AuthenticatedRequest, res: Response) => {
  try {
    const { caption, lyrics, bpm, duration, keyScale, timeSignature, temperature, topK, topP, lmModel, lmBackend } = req.body;

    if (!caption) {
      res.status(400).json({ error: 'Caption/style is required' });
      return;
    }

    const ACESTEP_API_URL = config.acestep.apiUrl;

    // Build param_obj for the REST API
    const paramObj: Record<string, unknown> = {};
    if (bpm && bpm > 0) paramObj.bpm = bpm;
    if (duration && duration > 0) paramObj.duration = duration;
    if (keyScale) paramObj.key = keyScale;
    if (timeSignature) paramObj.time_signature = timeSignature;

    // Primary path: call ACE-Step's /format_input REST endpoint (avoids Python spawn ENOENT on Windows)
    try {
      console.log(`[Format] Calling REST API: ${ACESTEP_API_URL}/format_input`);
      const apiRes = await fetch(`${ACESTEP_API_URL}/format_input`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          prompt: caption,
          lyrics: lyrics || '',
          temperature: temperature ?? 0.85,
          param_obj: paramObj,
        }),
        signal: AbortSignal.timeout(300_000), // 5 min — LLM may need to init first
      });

      const apiData = await apiRes.json() as any;

      if (!apiRes.ok || apiData.code !== 200) {
        const errMsg = apiData.error || apiData.detail || `Format API returned ${apiRes.status}`;
        console.error('[Format] API error:', errMsg);
        res.status(500).json({ success: false, error: errMsg });
        return;
      }

      const d = apiData.data;
      res.json({
        caption: d.caption,
        lyrics: d.lyrics,
        bpm: d.bpm,
        duration: d.duration,
        key_scale: d.key_scale,
        time_signature: d.time_signature,
        vocal_language: d.vocal_language,
      });
      return;
    } catch (fetchErr: any) {
      // Only fall back to Python spawn on network errors (service not yet reachable)
      if (fetchErr?.name !== 'AbortError' && (fetchErr?.code === 'ECONNREFUSED' || fetchErr?.cause?.code === 'ECONNREFUSED')) {
        console.warn('[Format] REST API unreachable, falling back to Python spawn');
      } else {
        console.error('[Format] REST API request failed:', fetchErr?.message);
        res.status(500).json({ success: false, error: fetchErr?.message || 'Format request failed' });
        return;
      }
    }

    // Fallback: Python spawn (only reached when REST API is unreachable)
    const { spawn } = await import('child_process');
    const ACESTEP_DIR = process.env.ACESTEP_PATH || path.resolve(path.dirname(fileURLToPath(import.meta.url)), '../../../ACE-Step-1.5');
    const __filename = fileURLToPath(import.meta.url);
    const __dirname = path.dirname(__filename);
    const SCRIPTS_DIR = path.join(__dirname, '../../scripts');
    const FORMAT_SCRIPT = path.join(SCRIPTS_DIR, 'format_sample.py');
    const pythonPath = resolvePythonPath(ACESTEP_DIR);

    const args = [FORMAT_SCRIPT, '--caption', caption, '--json'];
    if (lyrics) args.push('--lyrics', lyrics);
    if (bpm && bpm > 0) args.push('--bpm', String(bpm));
    if (duration && duration > 0) args.push('--duration', String(duration));
    if (keyScale) args.push('--key-scale', keyScale);
    if (timeSignature) args.push('--time-signature', timeSignature);
    if (temperature !== undefined) args.push('--temperature', String(temperature));
    if (topK && topK > 0) args.push('--top-k', String(topK));
    if (topP !== undefined) args.push('--top-p', String(topP));
    if (lmModel) args.push('--lm-model', lmModel);
    if (lmBackend) args.push('--lm-backend', lmBackend);

    console.log(`[Format] Fallback spawn: ${pythonPath} ${args.join(' ')}`);
    const result = await new Promise<{ success: boolean; data?: any; error?: string }>((resolve) => {
      const proc = spawn(pythonPath, args, {
        cwd: ACESTEP_DIR,
        env: { ...process.env, ACESTEP_PATH: ACESTEP_DIR },
      });

      let stdout = '';
      let stderr = '';

      proc.stdout.on('data', (data) => { stdout += data.toString(); });
      proc.stderr.on('data', (data) => { stderr += data.toString(); });

      proc.on('close', (code) => {
        if (code === 0 && stdout) {
          const lines = stdout.trim().split('\n');
          let jsonStr = '';
          for (let i = lines.length - 1; i >= 0; i--) {
            if (lines[i].startsWith('{')) { jsonStr = lines[i]; break; }
          }
          try {
            const parsed = JSON.parse(jsonStr || stdout);
            resolve({ success: true, data: parsed });
          } catch {
            console.error('[Format] Failed to parse stdout:', stdout.slice(0, 500));
            resolve({ success: false, error: 'Failed to parse format result' });
          }
        } else {
          console.error(`[Format] Process exited with code ${code}`);
          if (stdout) console.error('[Format] stdout:', stdout.slice(0, 1000));
          if (stderr) console.error('[Format] stderr:', stderr.slice(0, 1000));
          resolve({ success: false, error: stderr || stdout || `Format process exited with code ${code}` });
        }
      });

      proc.on('error', (err) => {
        console.error('[Format] Spawn error:', err.message);
        resolve({ success: false, error: err.message });
      });
    });

    if (result.success && result.data) {
      res.json(result.data);
    } else {
      console.error('[Format] Python error:', result.error);
      res.status(500).json({ success: false, error: result.error });
    }
  } catch (error) {
    console.error('[Format] Route error:', error);
    res.status(500).json({ error: (error as Error).message });
  }
});

export default router;
