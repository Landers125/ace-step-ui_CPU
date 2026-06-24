import { spawn } from 'child_process';
import { existsSync, mkdirSync } from 'fs';
import { homedir } from 'os';
import path from 'path';
import { fileURLToPath } from 'url';
import { pool } from '../db/pool.js';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const PROJECT_ROOT = path.resolve(__dirname, '../../..');
const CLOAKBROWSER_EXECUTABLE = path.join(PROJECT_ROOT, '.playwright-mcp/cloakbrowser-executable');
const FALLBACK_CLOAKBROWSER_EXECUTABLE = path.join(
  homedir(),
  '.cloakbrowser/chromium-145.0.7632.109.2/Chromium.app/Contents/MacOS/Chromium'
);

const CHROME_PATH = process.env.SUNO_CHROME_PATH ||
  (existsSync(CLOAKBROWSER_EXECUTABLE) ? CLOAKBROWSER_EXECUTABLE : FALLBACK_CLOAKBROWSER_EXECUTABLE);
const CHROME_CDP_PORT = process.env.SUNO_CHROME_CDP_PORT || '9323';
const CHROME_CDP_URL = process.env.SUNO_CHROME_CDP_URL || `http://127.0.0.1:${CHROME_CDP_PORT}`;
export const SUNO_CHROME_CDP_URL = CHROME_CDP_URL;
const CHROME_CDP_USER_DATA_DIR = process.env.SUNO_CHROME_CDP_PROFILE ||
  path.join(PROJECT_ROOT, '.playwright-mcp/profile');
const SUNO_START_URL = 'https://suno.com';
const CLERK_BASE_URL = 'https://clerk.suno.com';
const CLERK_VERSION = '5.15.0';
const JWT_REFRESH_LEEWAY_MS = 60 * 1000;

const COOKIE_ORIGINS = [
  'https://suno.com',
  'https://auth.suno.com',
  'https://studio-api.prod.suno.com',
  'https://studio-api-prod.suno.com',
];

const ESSENTIAL_COOKIE_NAMES = [
  '__client',
  '__client_uat',
  '__session',
  '__cf_bm',
  '_cfuvid',
  'ajs_anonymous_id',
];

interface SunoSettings {
  suno_cookie?: string | null;
  suno_sid?: string | null;
  suno_jwt?: string | null;
  suno_jwt_expires_at?: string | null;
}

interface JwtPayload {
  exp?: number;
  sub?: string;
  'suno.com/claims/email'?: string;
}

function shouldKeepCookie(name: string): boolean {
  return ESSENTIAL_COOKIE_NAMES.includes(name) ||
    name.startsWith('__client_') ||
    name.startsWith('__session_');
}

export function parseCookieString(cookieString: string): Record<string, string> {
  return cookieString
    .split(';')
    .map(part => part.trim())
    .filter(Boolean)
    .reduce<Record<string, string>>((cookies, part) => {
      const separator = part.indexOf('=');
      if (separator === -1) return cookies;
      const key = part.slice(0, separator).trim();
      const value = part.slice(separator + 1).trim();
      if (key) cookies[key] = value;
      return cookies;
    }, {});
}

function serializeCookieMap(cookies: Record<string, string>): string {
  return Object.entries(cookies)
    .filter(([name, value]) => shouldKeepCookie(name) && Boolean(value))
    .map(([name, value]) => `${name}=${value}`)
    .join('; ');
}

function decodeJwtPayload(token?: string): JwtPayload | null {
  if (!token || !token.includes('.')) return null;
  try {
    const payload = token.split('.')[1].replace(/-/g, '+').replace(/_/g, '/');
    const padded = payload.padEnd(payload.length + ((4 - payload.length % 4) % 4), '=');
    return JSON.parse(Buffer.from(padded, 'base64').toString('utf8'));
  } catch {
    return null;
  }
}

function getClientToken(cookies: Record<string, string>): string | undefined {
  return cookies.__client ||
    Object.entries(cookies).find(([key]) => key.startsWith('__client') && !key.includes('_uat'))?.[1];
}

export function getCookieStatus(cookieString?: string | null) {
  const cookies = parseCookieString(cookieString || '');
  const sessionPayload = decodeJwtPayload(cookies.__session);
  const expiresAt = sessionPayload?.exp ? new Date(sessionPayload.exp * 1000).toISOString() : null;
  const isExpired = sessionPayload?.exp ? sessionPayload.exp * 1000 <= Date.now() : null;

  return {
    hasCookie: Boolean(cookieString),
    hasClientCookie: Boolean(getClientToken(cookies)),
    hasSessionCookie: Boolean(cookies.__session),
    isSessionValid: isExpired === null ? null : !isExpired,
    expiresAt,
    userId: sessionPayload?.sub || null,
    email: sessionPayload?.['suno.com/claims/email'] || null,
  };
}

export async function getSunoSettings(): Promise<SunoSettings> {
  const result = await pool.query(
    'SELECT suno_cookie, suno_sid, suno_jwt, suno_jwt_expires_at FROM settings WHERE id = ?',
    ['default']
  );
  return result.rows[0] || {};
}

async function upsertSunoSettings(values: Partial<SunoSettings>): Promise<void> {
  await pool.query(
    `INSERT INTO settings (id, suno_cookie, suno_sid, suno_jwt, suno_jwt_expires_at, created_at, updated_at)
     VALUES (?, ?, ?, ?, ?, datetime('now'), datetime('now'))
     ON CONFLICT(id) DO UPDATE SET
       suno_cookie = COALESCE(excluded.suno_cookie, settings.suno_cookie),
       suno_sid = COALESCE(excluded.suno_sid, settings.suno_sid),
       suno_jwt = COALESCE(excluded.suno_jwt, settings.suno_jwt),
       suno_jwt_expires_at = COALESCE(excluded.suno_jwt_expires_at, settings.suno_jwt_expires_at),
       updated_at = datetime('now')`,
    [
      'default',
      values.suno_cookie ?? null,
      values.suno_sid ?? null,
      values.suno_jwt ?? null,
      values.suno_jwt_expires_at ?? null,
    ]
  );
}

export async function saveSunoCookie(cookieString: string) {
  const filtered = serializeCookieMap(parseCookieString(cookieString));
  if (!filtered) {
    throw new Error('No supported Suno cookies were found');
  }

  await pool.query(
    `INSERT INTO settings (id, suno_cookie, suno_sid, suno_jwt, suno_jwt_expires_at, created_at, updated_at)
     VALUES (?, ?, NULL, NULL, NULL, datetime('now'), datetime('now'))
     ON CONFLICT(id) DO UPDATE SET
       suno_cookie = excluded.suno_cookie,
       suno_sid = NULL,
       suno_jwt = NULL,
       suno_jwt_expires_at = NULL,
       updated_at = datetime('now')`,
    ['default', filtered]
  );

  return getCookieStatus(filtered);
}

export async function getCdpStatus() {
  try {
    const response = await fetch(`${CHROME_CDP_URL}/json/version`);
    if (!response.ok) {
      return { running: false, cdpUrl: CHROME_CDP_URL };
    }
    const version = await response.json().catch(() => ({}));
    return { running: true, cdpUrl: CHROME_CDP_URL, version };
  } catch {
    return { running: false, cdpUrl: CHROME_CDP_URL };
  }
}

async function openSunoPageInCdp(): Promise<boolean> {
  try {
    const response = await fetch(`${CHROME_CDP_URL}/json/new?${encodeURIComponent(SUNO_START_URL)}`, {
      method: 'PUT',
    });
    return response.ok;
  } catch {
    return false;
  }
}

function activateChromeBestEffort(): void {
  spawn('osascript', ['-e', 'tell application "Chromium" to activate'], {
    detached: true,
    stdio: 'ignore',
  }).unref();
}

export async function launchChromeCdp() {
  const status = await getCdpStatus();
  if (status.running) {
    await openSunoPageInCdp();
    activateChromeBestEffort();
    return { ...status, launched: false, openedSuno: true };
  }
  if (!existsSync(CHROME_PATH)) {
    throw new Error(`Chrome was not found at ${CHROME_PATH}`);
  }

  mkdirSync(CHROME_CDP_USER_DATA_DIR, { recursive: true });

  const child = spawn(CHROME_PATH, [
    `--remote-debugging-port=${CHROME_CDP_PORT}`,
    `--user-data-dir=${CHROME_CDP_USER_DATA_DIR}`,
    '--fingerprint=10526',
    '--fingerprint-platform=macos',
    '--password-store=basic',
    '--use-mock-keychain',
    '--no-first-run',
    '--no-default-browser-check',
    '--disable-features=DestroyProfileOnBrowserClose',
    SUNO_START_URL,
  ], {
    detached: true,
    stdio: 'ignore',
  });
  child.unref();
  activateChromeBestEffort();

  for (let i = 0; i < 20; i += 1) {
    await new Promise(resolve => setTimeout(resolve, 250));
    const next = await getCdpStatus();
    if (next.running) return { ...next, launched: true };
  }

  return { running: false, launched: true, cdpUrl: CHROME_CDP_URL };
}

export async function extractSunoCookiesFromChrome() {
  const { chromium } = await import('rebrowser-playwright-core');
  const browser = await chromium.connectOverCDP(CHROME_CDP_URL);
  try {
    const contexts = browser.contexts();
    if (contexts.length === 0) {
      throw new Error('No Chrome context is available. Open the Suno login Chrome first.');
    }

    const cookieMap: Record<string, string> = {};
    for (const context of contexts) {
      const cookies = await context.cookies(COOKIE_ORIGINS);
      for (const cookie of cookies) {
        if (shouldKeepCookie(cookie.name)) {
          cookieMap[cookie.name] = cookie.value;
        }
      }
    }

    const cookieString = serializeCookieMap(cookieMap);
    if (!cookieString) {
      throw new Error('No Suno cookies were found. Log in to suno.com in the dedicated Chrome window first.');
    }

    await saveSunoCookie(cookieString);
    return getCookieStatus(cookieString);
  } finally {
    await (browser as any).disconnect?.().catch(() => undefined);
  }
}

async function getSessionId(cookieString: string): Promise<string> {
  const cookies = parseCookieString(cookieString);
  const clientToken = getClientToken(cookies);
  if (!clientToken) {
    throw new Error('No __client cookie found. Re-import Suno cookies from Chrome.');
  }

  const response = await fetch(`${CLERK_BASE_URL}/v1/client?_is_native=true&_clerk_js_version=${CLERK_VERSION}`, {
    headers: {
      Authorization: clientToken,
      Cookie: cookieString,
    },
  });
  const data = await response.json().catch(() => ({}));
  const sid = data?.response?.last_active_session_id;
  if (!response.ok || !sid) {
    throw new Error('Failed to get Suno session id. Re-login to Suno and import cookies again.');
  }
  return sid;
}

export async function refreshSunoJwt(cookieCandidate?: string | null) {
  const settings = await getSunoSettings();
  const cookieString = cookieCandidate || settings.suno_cookie || process.env.SUNO_COOKIE;
  if (!cookieString) {
    throw new Error('Suno cookie is not configured. Open Suno login Chrome and import cookies first.');
  }

  const cookies = parseCookieString(cookieString);
  const clientToken = getClientToken(cookies);
  if (!clientToken) {
    throw new Error('No __client cookie found. Re-import Suno cookies from Chrome.');
  }

  const sid = settings.suno_sid || await getSessionId(cookieString);
  const response = await fetch(
    `${CLERK_BASE_URL}/v1/client/sessions/${encodeURIComponent(sid)}/tokens?_is_native=true&_clerk_js_version=${CLERK_VERSION}`,
    {
      method: 'POST',
      headers: {
        Authorization: clientToken,
        Cookie: cookieString,
        'Content-Type': 'application/json',
      },
      body: '{}',
    }
  );
  const data = await response.json().catch(() => ({}));
  const jwt = data?.jwt;
  if (!response.ok || !jwt) {
    throw new Error('Failed to refresh Suno JWT. Re-login to Suno and import cookies again.');
  }

  const payload = decodeJwtPayload(jwt);
  const expiresAt = payload?.exp ? new Date(payload.exp * 1000).toISOString() : null;
  await upsertSunoSettings({
    suno_cookie: cookieString,
    suno_sid: sid,
    suno_jwt: jwt,
    suno_jwt_expires_at: expiresAt,
  });

  return {
    jwt,
    expiresAt,
    userId: payload?.sub || null,
    email: payload?.['suno.com/claims/email'] || null,
  };
}

export async function getSunoBearerToken(overrideToken?: string | null): Promise<string> {
  if (overrideToken?.trim()) return overrideToken.trim();

  const settings = await getSunoSettings();
  if (settings.suno_jwt && settings.suno_jwt_expires_at) {
    const expiresAt = new Date(settings.suno_jwt_expires_at).getTime();
    if (Number.isFinite(expiresAt) && expiresAt - JWT_REFRESH_LEEWAY_MS > Date.now()) {
      return settings.suno_jwt;
    }
  }

  try {
    const refreshed = await refreshSunoJwt(settings.suno_cookie || process.env.SUNO_COOKIE || null);
    return refreshed.jwt;
  } catch (firstError) {
    await launchChromeCdp();
    await extractSunoCookiesFromChrome();
    try {
      const refreshed = await refreshSunoJwt();
      return refreshed.jwt;
    } catch {
      throw firstError;
    }
  }
}

export async function getSunoAuthStatus() {
  const settings = await getSunoSettings();
  return {
    ...getCookieStatus(settings.suno_cookie || process.env.SUNO_COOKIE || null),
    hasJwt: Boolean(settings.suno_jwt),
    jwtExpiresAt: settings.suno_jwt_expires_at || null,
    cdp: await getCdpStatus(),
  };
}
