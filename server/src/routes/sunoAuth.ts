import { Router, Response } from 'express';
import { authMiddleware, AuthenticatedRequest } from '../middleware/auth.js';
import {
  extractSunoCookiesFromChrome,
  getCdpStatus,
  getSunoAuthStatus,
  launchChromeCdp,
  refreshSunoJwt,
  saveSunoCookie,
} from '../services/sunoAuth.js';

const router = Router();

router.get('/chrome-cdp', authMiddleware, async (_req: AuthenticatedRequest, res: Response) => {
  try {
    res.json(await getCdpStatus());
  } catch (error) {
    res.status(500).json({ error: (error as Error).message || 'Failed to check Chrome CDP status' });
  }
});

router.post('/chrome-cdp', authMiddleware, async (_req: AuthenticatedRequest, res: Response) => {
  try {
    res.json(await launchChromeCdp());
  } catch (error) {
    res.status(500).json({ error: (error as Error).message || 'Failed to launch Chrome CDP' });
  }
});

router.get('/login', authMiddleware, async (_req: AuthenticatedRequest, res: Response) => {
  try {
    res.json(await getSunoAuthStatus());
  } catch (error) {
    res.status(500).json({ error: (error as Error).message || 'Failed to read Suno auth status' });
  }
});

router.post('/login', authMiddleware, async (_req: AuthenticatedRequest, res: Response) => {
  try {
    const status = await extractSunoCookiesFromChrome();
    res.json({ success: true, ...status });
  } catch (error) {
    res.status(500).json({ error: (error as Error).message || 'Failed to import Suno cookies from Chrome' });
  }
});

router.post('/cookie', authMiddleware, async (req: AuthenticatedRequest, res: Response) => {
  try {
    const { cookie } = req.body as { cookie?: string };
    if (!cookie?.trim()) {
      res.status(400).json({ error: 'Cookie is required' });
      return;
    }
    const status = await saveSunoCookie(cookie.trim());
    res.json({ success: true, ...status });
  } catch (error) {
    res.status(500).json({ error: (error as Error).message || 'Failed to save Suno cookie' });
  }
});

router.post('/refresh', authMiddleware, async (_req: AuthenticatedRequest, res: Response) => {
  try {
    const token = await refreshSunoJwt();
    res.json({
      success: true,
      hasJwt: Boolean(token.jwt),
      expiresAt: token.expiresAt,
      userId: token.userId,
      email: token.email,
    });
  } catch (error) {
    res.status(500).json({ error: (error as Error).message || 'Failed to refresh Suno JWT' });
  }
});

export default router;
