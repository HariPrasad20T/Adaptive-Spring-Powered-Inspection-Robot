function SpringRobot_V7_SpringJumper()
% Adaptive spring-legged JUMPING inspection robot - digital twin V7 (3D)
% Four telescopic coil-spring legs (guide sleeve + plunger + visible coil), spring-loaded jump
% CHANGE: jumps are now launched ON THE MOVE (run-up).  The robot never stops before a jump:
%         it keeps walking at p.vJump, charges the leg springs while moving and the spring only
%         adds the missing velocity (dvx = vx - vRunUp, dvz = vz).
close all; clc; rng(7);
opts.capMap    = true;
opts.export    = true;
opts.ai        = true;
opts.dynamic   = true;
opts.rocks     = true;          % moving rocks that roll across the lane
opts.faults    = true;
opts.routeMode = 'safest';
opts.adaptive  = true;
opts.multi     = false;
opts.nRobots   = 3;
opts.terrain   = true;
opts.collision = true;
p   = robotParams();
env = envParams();
if ~opts.rocks, env.dyn = env.dyn(~[env.dyn.rock]); end
env.aiOn = opts.ai;  env.dynOn = opts.dynamic;  env.faultsOn = opts.faults;  env.routeMode = opts.routeMode;
env.asOn = opts.adaptive;  env.multiOn = opts.multi;  env.nRobots = opts.nRobots;
env.terrOn = opts.terrain;  env.colOn = opts.collision;
printHeader(p, env);
fprintf(' Training AI decision network ...\n');
net = trainDecisionNet(p);
fprintf(' AI network ready: %d training samples, validation accuracy %.1f %%\n', net.nTrain, 100*net.valAcc);
S = runSimulation(p, env, net);
S.MC = monteCarlo(p, env, S);
S.D  = buildSensorLog(p, env, S);
S.T = [];  S.C = [];  S.tcls = [];
if env.terrOn
    S.tcls = trainTerrainClassifier(p);
    S.T = buildTerrainLog(p, env, S, S.tcls);
    fprintf(' Terrain classifier trained, online accuracy %.1f %%\n', 100*mean(S.T(:,1) == S.T(:,2)));
end
if env.colOn, S.C = buildCollisionLog(p, env, S); end
S.HZ = hazardRanking(p, env, S);
S.EF = energyForecast(p, env, S);
S.M  = [];
if env.multiOn, S.M = runTeam(p, env, net, S); end
CM = [];
if opts.capMap
    fprintf(' Computing jump capability map ...\n');
    CM = capabilityMap(p, env);
end
printSummary(p, env, S, CM);
printTC(p, env, S);
printNew(p, env, S);
printExtra(p, env, S);
playAnimation(p, env, S);
f2 = showAnalytics(p, env, S);
f3 = showAdvanced(p, env, S, CM);
f4 = showExtensions(p, env, S, net);
f5 = showSensorLog(p, env, S);
f6 = showAdaptive(p, env, S);
f8 = showTerrainCollision(p, env, S);
f7 = showTeam(p, env, S);
f9 = showHazardEnergy(p, env, S);
if opts.export
    exportReport(p, env, S, CM, [f2, f3, f4, f5, f6, f8, f7, f9]);
end
end
%% =====================================================================
%%  PARAMETERS
%% =====================================================================
function p = robotParams()
p.g  = 9.81;
p.m  = 1.8;
p.L  = 0.30;  p.H = 0.08;
p.W  = 0.22;                       % chassis width (m)
p.wb = 0.20;                       % fore-aft distance between front and hind hip axes (m)
p.r  = 0.05;                       % effective foot-swing envelope radius (m): sets the step-over limit
p.wy = 0.125;                      % lateral hip / foot offset (m)
p.ww = 0.04;
p.hw = 0.07;                       % COM -> foot-envelope centre (m)
p.I  = p.m*(p.L^2 + p.H^2)/12;
p.restZ   = p.r + p.hw;
p.legRest = 0.075;
p.cornerX = [-p.L/2,  p.L/2, p.L/2, -p.L/2];
p.cornerZ = [-p.H/2, -p.H/2, p.H/2,  p.H/2];
% ---- spring-leg model (V7) ----
p.legL     = 0.045;                % guide sleeve length of each telescopic leg (m), drawing only
p.rf       = 0.020;                % foot pad radius (m), drawing only
p.coilR    = 0.016;                % coil spring radius (m), drawing only
p.nCoil    = 7;                    % number of visible coil turns, drawing only
p.stride   = 0.16;                 % gait cycle length (m)
p.stepH    = 0.045;                % swing foot lift (m)
p.gaitDuty = 0.50;                 % stance fraction (trot)
p.bobAmp   = 0.010;                % body bounce on the leg springs while walking (m), drawing only
p.k           = 4500;              % elastic leg-tendon stiffness (N/m), all legs combined
p.eta         = 0.85;
p.etaWinch    = 0.80;
p.maxTravel   = 0.08;
p.thrustOffset= 0.004;
p.armSR       = 0.06;
p.Kp = 1.0;  p.Kd = 0.18;  p.tauMax = 0.25;
p.vCruise    = 0.8;
% ---- running jump (NEW) ----
p.vJump      = 0.6;                % forward run-up speed kept during the jump (m/s): the robot does NOT stop
p.nCrouchRun = 25;                 % crouch frames (10 ms each) while running (0.25 s)
p.etaDrive   = 0.70;
p.hRollMax   = 0.6*p.r;            % step-over limit: highest obstacle the legs can walk over (m)
p.senseRange = 0.90;
p.senseNoise = 0.002;
p.safetyMargin = 0.03;
p.landMargin   = 0.05;
p.phiLimit     = 20*pi/180;
p.vzLimit      = 4.0;
p.tipReserve   = 0.15;
p.nScan    = 8;
p.kSigma   = 2;
p.fatigue  = 0.02;
p.healthMin= 0.70;
p.Ebat     = 220;
p.solarW   = 0.5;
p.mcN      = 300;
p.mcSigV   = 0.03;
p.mcSigTh  = 1.5*pi/180;
p.mcSigD   = 0.010;
p.dynRange  = 2.2;
p.dynNoise  = 0.01;
p.laneLimit = 0.85;
p.infl      = 0.18;
p.robotHW   = 0.145;
p.aiConf    = 0.60;
end
function env = envParams()
env.xk = [0   3.5  5.0  8.0  9.5  19   20.5 22.5 24   33];
env.zk = [0   0    0.12 0.12 0    0    0.12 0.12 0    0 ];
env.zone.name  = {'Asphalt','Ramp up','Gravel','Ramp down','Grass','Sand', ...
                  'Asphalt','Ramp up','Gravel','Ramp down','Grass','Sand'};
env.zone.x0    = [0   3.5  5.0  8.0  9.5  12.5  17   19   20.5 22.5 24   27];
env.zone.x1    = [3.5 5.0  8.0  9.5  12.5 17    19   20.5 22.5 24   27   33];
env.zone.crr   = [0.02 0.03 0.06 0.03 0.08 0.15  0.02 0.03 0.06 0.03 0.08 0.15];
env.zone.soft  = [1.00 1.00 0.97 1.00 0.95 0.88  1.00 1.00 0.97 1.00 0.95 0.88];
zc6 = [0.30 0.30 0.33; 0.42 0.42 0.45; 0.58 0.53 0.46; 0.42 0.42 0.45; 0.36 0.58 0.26; 0.88 0.77 0.50];
env.zone.color = [zc6; zc6];
env.obs     = [ 1.00 0.20 0.020;
                2.60 0.18 0.080;
                6.00 0.20 0.120;
               10.50 0.25 0.150;
               13.30 0.15 0.100;
               15.20 0.20 0.025;
               17.80 0.20 0.100;
               21.30 0.15 0.090;
               25.00 0.30 0.140;
               26.60 0.10 0.020;
               28.00 0.22 0.120;
               29.30 0.18 0.060];
env.obsName = {'curb','kerb step','rock slab','concrete block','log','speed bump','pipe', ...
               'sandbag wall','fallen tree','cable cover','boulder','drain grate'};
env.obsHalfW = [0.90 0.90 0.30 0.90 0.50 0.90 0.90 0.35 0.90 0.15 0.45 0.60];
env.xEnd    = 30.5;
env.kick.x     = 14.4;
env.kick.omega = -9;

% 11 moving objects: 8 rocks (rock = true) and 3 carts / vehicles.
env.dyn = struct('name',  {'rolling rock', 'crossing cart', 'patrol rover', 'survey cart', 'tumbling boulder', 'rock slide', 'tractor', ...
                           'rolling rock B', 'tumbling rock', 'rock slide B', 'boulder roll'}, ...
                 'x',     {4.20, 7.40, 12.00, 16.20, 17.00, 23.40, 27.40, 2.00, 9.00, 19.80, 29.90}, ...
                 'len',   {0.30, 0.30, 0.30, 0.30, 0.30, 0.30, 0.30, 0.30, 0.30, 0.30, 0.30}, ...
                 'hw',    {0.15, 0.20, 0.20, 0.20, 0.15, 0.15, 0.20, 0.14, 0.16, 0.15, 0.17}, ...
                 'y0',    {1.25, -1.25, 1.25, 1.25, -1.25, 1.25, -1.25, -1.25, 1.25, -1.25, 1.25}, ...
                 'vy',    {-0.40, 0.45, -0.35, -0.50, 0.42, -0.38, 0.40, 0.44, -0.41, 0.46, -0.43}, ...
                 'trigX', {2.90, 5.10, 10.90, 15.10, 15.90, 22.00, 26.00, 0.80, 7.00, 18.00, 28.00}, ...
                 'rock',  {true, false, false, false, true, true, false, true, true, true, true});

env.faults = struct('type', {'LIDAR','SPRING','IMU','MOTOR','SPRING','LIDAR','MOTOR','IMU','SPRING'}, ...
                    'x0',   {0.30, 1.40, 5.20, 8.60, 14.00, 17.00, 20.60, 24.50, 28.40}, ...
                    'x1',   {2.20, 3.20, 6.60, 9.90, 15.50, 18.60, 22.20, 26.00, 29.80}, ...
                    'a',    {6,    0.15, 0.60, 0.55, 0.20,  5,     0.60,  0.55,  0.25}, ...
                    'b',    {0.75, 0,    0,    0.70, 0,     0.80,  0.75,  0,     0});
env.aiOn = true;  env.dynOn = true;  env.faultsOn = true;  env.routeMode = 'safest';
env.asOn = true;  env.multiOn = true;  env.nRobots = 3;
env.terrOn = true;  env.colOn = true;
end

%% =====================================================================
%%  TERRAIN HELPERS
%% =====================================================================
function z = terrainZ(env, x)
z = interp1(env.xk, env.zk, x, 'linear', 'extrap');
end

function z = groundHeight(env, x)
z = terrainZ(env, x);
for i = 1:size(env.obs,1)
    in = (x >= env.obs(i,1)) & (x <= env.obs(i,1) + env.obs(i,2));
    z(in) = z(in) + env.obs(i,3);
end
end

function z = groundHeightXY(env, X, Y)
z = terrainZ(env, X);
for i = 1:size(env.obs,1)
    in = (X >= env.obs(i,1)) & (X <= env.obs(i,1) + env.obs(i,2)) & (abs(Y) <= env.obsHalfW(i));
    z(in) = z(in) + env.obs(i,3);
end
end

function e2 = envAtY(env, y)
e2 = env;
keep = abs(y) < env.obsHalfW(:)' + 0.17;
e2.obs = env.obs(keep, :);
end

function P = curPose(newP)
persistent pp
if isempty(pp), pp = [0 0]; end
if nargin > 0, pp = newP; end
P = pp;
end

function g = laneGoal(newG)
persistent gg
if isempty(gg), gg = 0; end
if nargin > 0, gg = newG; end
g = gg;
end

function zi = zoneAt(env, x)
zi = find(x >= env.zone.x0, 1, 'last');
if isempty(zi), zi = 1; end
end

function ok = isFlat(env, xa, xb, zg, iSkip)
xs = linspace(xa, xb, 40);
ok = all(abs(terrainZ(env, xs) - zg) < 1e-6);
for j = 1:size(env.obs,1)
    if j ~= iSkip && env.obs(j,1) < xb && (env.obs(j,1) + env.obs(j,2)) > xa
        ok = false;
    end
end
end

function zw = wheelCenterZ(p, env, xw)
% height of the foot-envelope centre over the terrain under one leg pair
xs = linspace(xw - p.r, xw + p.r, 41);
zs = groundHeight(env, xs) + sqrt(max(p.r^2 - (xs - xw).^2, 0));
zw = max(zs);
end

function [zc, phi] = rollPose(p, env, x)
zf  = wheelCenterZ(p, env, x + p.wb/2);
zr  = wheelCenterZ(p, env, x - p.wb/2);
phi = atan2(zf - zr, p.wb);
zc  = 0.5*(zf + zr) + p.hw*cos(phi);
end

function a = wrapPi(a)
a = mod(a + pi, 2*pi) - pi;
end

function [X, Z] = toWorld(bx, bz, x, z, phi)
X = x + bx*cos(phi) - bz*sin(phi);
Z = z + bx*sin(phi) + bz*cos(phi);
end

function [dx, dz, which] = lowestPoint(p, phi)
c = cos(phi);  s = sin(phi);
bxw = [-p.wb/2, p.wb/2];
wx  = bxw*c + p.hw*s;
wz  = bxw*s - p.hw*c - p.r;
cx  = p.cornerX*c - p.cornerZ*s;
cz  = p.cornerX*s + p.cornerZ*c;
X = [wx, cx];  Z = [wz, cz];
[dz, i] = min(Z);
dx = X(i);
which = 1 + (i > 2);
end

function soc = batterySOC(p, t, E)
soc = min(max(1 - (E - p.solarW*t)/p.Ebat, 0), 1);
end

%% =====================================================================
%%  RUNNING-JUMP HELPERS (NEW)
%% =====================================================================
function ve = springSpeed(p, v, th)
% velocity that the leg springs must ADD so that the world launch velocity (v, th) is reached
% while the robot is already running forward at min(p.vJump, vx).
vx = v.*cos(th);  vz = v.*sin(th);
va = min(p.vJump, vx);
ve = sqrt(vz.^2 + (vx - va).^2);
end

function v = launchFromSpring(p, ve, th)
% inverse of springSpeed: world launch speed obtained from a spring-delivered speed ve at angle th
s = sin(th);  c = cos(th);  vJ = p.vJump;
v = ve/s;
if v*c > vJ
    v = c*vJ + sqrt(max(ve^2 - (s*vJ)^2, 0));
end
end

function trv = launchTravel(vApp, vxL, nC, tp)
% horizontal distance covered by the robot from the start of the crouch to the release point
nT = 6;  dtE = tp/nT;
trv = vApp*0.01*nC;
for j = 1:nT
    trv = trv + (vApp + (vxL - vApp)*(j/nT))*dtE;
end
end

%% =====================================================================
%%  PHYSICS: FLIGHT, LANDING, TUMBLE, JUMP PLANNER, SELF-RIGHTING PLANNER
%% =====================================================================
function fl = simulateFlight(p, z0, vx, vz0, w0, ctrlOn, zg, phi0)
dt = 0.0025;  nMax = 2400;
T = zeros(nMax,1); X = T; Z = T; PH = T; W = T; VZ = T; TAU = T;
x = 0; z = z0; phi = phi0; w = w0; vz = vz0;
n = 1;  Z(1) = z; PH(1) = phi; W(1) = w; VZ(1) = vz;
hit = false;
while n < nMax
    tau = 0;
    if ctrlOn
        tau = min(max(-p.Kp*phi - p.Kd*w, -p.tauMax), p.tauMax);
    end
    w   = w + tau/p.I*dt;
    phi = phi + w*dt;
    vz  = vz - p.g*dt;
    x   = x + vx*dt;
    z   = z + vz*dt;
    n   = n + 1;
    T(n) = (n-1)*dt; X(n) = x; Z(n) = z; PH(n) = phi; W(n) = w; VZ(n) = vz; TAU(n) = tau;
    [~, dz] = lowestPoint(p, phi);
    if vz < 0 && (z + dz) <= zg
        hit = true;
        break;
    end
end
fl.hit = hit;  fl.n = n;
fl.t = T(1:n); fl.x = X(1:n); fl.z = Z(1:n); fl.phi = PH(1:n); fl.w = W(1:n); fl.vz = VZ(1:n);
fl.vx = vx;
fl.xTD = x; fl.zTD = z; fl.vxTD = vx; fl.vzTD = vz; fl.wTD = w; fl.phiTD = phi; fl.T = T(n);
fl.Hrw     = sum(TAU(1:n))*dt;
fl.tauPeak = max(abs(TAU(1:n)));
fl.apex    = max(Z(1:n)) - zg;
end

function lc = landingCheck(p, vx, vz, w, phi)
lc.phi      = phi;
lc.pitchOK  = abs(wrapPi(phi)) <= p.phiLimit;
lc.speedOK  = abs(vz) <= p.vzLimit;
c = cos(phi); s = sin(phi);
margin = 1; Hmax = 0;
for bx = [-p.wb/2, p.wb/2]
    px = bx*c + p.hw*s;
    pz = bx*s - p.hw*c - p.r;
    rx = -px;  rz = -pz;
    HP = p.I*w + p.m*(rx*vz - rz*vx);
    IP = p.I + p.m*(rx^2 + rz^2);
    wp = HP/IP;
    Ek   = 0.5*IP*wp^2;
    Ebar = p.m*p.g*(hypot(rx, rz) - rz);
    Hmax = max(Hmax, abs(HP));
    if wp*rx > 0
        margin = min(margin, 1 - Ek/Ebar);
    end
end
lc.tipMargin = margin;
lc.tipOK     = margin > p.tipReserve;
lc.Hmax      = Hmax;
lc.stable    = lc.pitchOK && lc.speedOK && lc.tipOK;
end

function tm = tumbleSim(p, r0, w0)
dt = 0.001;  nMax = 4000;
IP = p.I + p.m*(r0(1)^2 + r0(2)^2);
dir = sign(w0);
T = zeros(nMax,1); D = T;
dphi = 0; w = w0; n = 1; inverted = false;
while n < nMax
    rx  = cos(dphi)*r0(1) - sin(dphi)*r0(2);
    tau = -p.m*p.g*rx - 0.02*w;
    w    = w + tau/IP*dt;
    dphi = dphi + w*dt;
    n = n + 1;
    T(n) = (n-1)*dt;
    if dir*dphi >= pi
        dphi = dir*pi;  D(n) = dphi;  inverted = true;  break;
    end
    D(n) = dphi;
    if dir*w <= 0, break; end
end
tm.t = T(1:n);  tm.dphi = D(1:n);  tm.inverted = inverted;
tm.IP = IP;  tm.Ek = 0.5*IP*w0^2;  tm.Ebar = p.m*p.g*(hypot(r0(1), r0(2)) - r0(2));
end

function mc = minClearance(p, ob, X, Z, PHI, zg)
xs = ob(1);  xe = ob(1) + ob(2);  top = zg + ob(3);
mc = Inf;
for bx = [-p.wb/2, p.wb/2]
    cx = X + bx.*cos(PHI) + p.hw*sin(PHI);
    cz = Z + bx.*sin(PHI) - p.hw*cos(PHI);
    inside = (cx >= xs - p.r) & (cx <= xe + p.r);
    if any(inside)
        mc = min(mc, min(cz(inside) - p.r - top));
    end
end
end

function plan = planJump(p, env, iObs, hM, wM, zg, soil)
g = p.g;
plan = struct('feasible',false,'nCand',0,'rejSpring',0,'rejFlat',0,'rejClear',0,'rejStab',0, ...
              'v',NaN,'theta',NaN,'d1',NaN,'comp',NaN,'force',NaN,'Espring',NaN,'Ecost',NaN, ...
              'vx',NaN,'vz',NaN,'xLaunch',NaN,'fl',[],'lc',[],'mc',NaN, ...
              'phiU',NaN,'phiC',NaN,'arcX',[],'arcZ',[],'vmax',NaN,'vApp',NaN);
xObs = env.obs(iObs,1);
etaE = p.eta*soil;
% vmax = largest velocity the leg springs can ADD (spring-delivered speed)
plan.vmax = sqrt(max(etaE*p.k*p.maxTravel^2 - 2*p.m*g*p.maxTravel, 0)/p.m);
hc = hM + p.safetyMargin;  hb = p.wb/2;

cand = zeros(0,4);
for th = (40:5:80)*pi/180
    for d1 = 0.15:0.05:0.45
        d2 = d1 + wM;  xa = d1 - hb;  xb = d2 + hb;
        if xa*tan(th) - hc <= 0, continue; end
        vA = g*xa^2/(2*cos(th)^2*(xa*tan(th) - hc));
        vB = g*xb^2/(2*cos(th)^2*(xb*tan(th) - hc));
        vR = g*(d2 + hb + p.landMargin)/sin(2*th);
        v0 = sqrt(max([vA, vB, vR]));
        cand(end+1,:) = [v0, th, d1, springSpeed(p, v0, th)]; %#ok<AGROW>
    end
end
if isempty(cand), return; end
cand = sortrows(cand, 4);                 % cheapest SPRING effort first
plan.nCand = size(cand,1);

for c = 1:size(cand,1)
    v = cand(c,1); th = cand(c,2); d1 = cand(c,3); ve = cand(c,4);
    if ve > plan.vmax
        plan.rejSpring = plan.rejSpring + (size(cand,1) - c + 1);
        break;
    end
    xl   = xObs - d1;
    comp = (p.m*g + sqrt((p.m*g)^2 + etaE*p.k*p.m*ve^2))/(etaE*p.k);
    w0   = p.m*v*sin(th)*p.thrustOffset/p.I;
    fl   = simulateFlight(p, zg + p.restZ + comp, v*cos(th), v*sin(th), w0, true, zg, 0);
    if ~fl.hit, continue; end
    if ~isFlat(env, xl - p.L/2, xl + fl.xTD + hb + 0.2, zg, iObs)
        plan.rejFlat = plan.rejFlat + 1;  continue;
    end
    mc = minClearance(p, [xObs, wM, hM], xl + fl.x, fl.z, fl.phi, zg);
    if mc < 0.01
        plan.rejClear = plan.rejClear + 1;  continue;
    end
    lc = landingCheck(p, fl.vxTD, fl.vzTD, fl.wTD, fl.phiTD);
    if ~lc.stable
        plan.rejStab = plan.rejStab + 1;  continue;
    end
    flU = simulateFlight(p, zg + p.restZ + comp, v*cos(th), v*sin(th), w0, false, zg, 0);
    plan.feasible = true;
    plan.v = v; plan.theta = th; plan.d1 = d1; plan.comp = comp; plan.force = p.k*comp;
    plan.Espring = 0.5*p.k*comp^2;  plan.Ecost = plan.Espring/p.etaWinch;
    plan.vx = v*cos(th); plan.vz = v*sin(th); plan.xLaunch = xl;
    plan.vApp = min(p.vJump, plan.vx);
    plan.fl = fl; plan.lc = lc; plan.mc = mc;
    plan.phiC = wrapPi(fl.phiTD)*180/pi;  plan.phiU = wrapPi(flU.phiTD)*180/pi;
    plan.arcX = xl + fl.x;  plan.arcZ = fl.z;
    break;
end
end

function sr = planSelfRight(p, soil, zg)
etaE = p.eta*soil;  d = p.armSR;
comp = 0.04;
for it = 1:4
    z0 = zg + p.H/2 + comp;
    lo = 0.8;  hi = 2.5;
    for j = 1:35
        vz0 = 0.5*(lo + hi);
        w0  = p.m*vz0*d/p.I;
        fl  = simulateFlight(p, z0, 0, vz0, w0, false, zg, -pi);
        if fl.phiTD < 0, lo = vz0; else, hi = vz0; end
    end
    vz0  = 0.5*(lo + hi);
    w0   = p.m*vz0*d/p.I;
    comp = (p.m*p.g + sqrt((p.m*p.g)^2 + etaE*p.k*(p.m*vz0^2 + p.I*w0^2)))/(etaE*p.k);
end
z0 = zg + p.H/2 + comp;
fl = simulateFlight(p, z0, 0, vz0, w0, false, zg, -pi);
sr.vz = vz0; sr.w = w0; sr.L = p.I*w0; sr.comp = comp; sr.force = p.k*comp;
sr.Ecost = 0.5*p.k*comp^2/p.etaWinch;  sr.tp = 2*comp/vz0;  sr.fl = fl;
sr.feasible = fl.hit && comp <= p.maxTravel && abs(wrapPi(fl.phiTD)) < 10*pi/180;
end

%% =====================================================================
%%  SENSOR FUSION, MONTE-CARLO RISK, CAPABILITY MAP
%% =====================================================================
function [xe, sig] = fuseScans(truth, R, n)
z  = truth + R*randn(1, n);
xe = z(1);  P = R^2;
for k = 2:n
    K  = P/(P + R^2);
    xe = xe + K*(z(k) - xe);
    P  = (1 - K)*P;
end
sig = sqrt(P);
end

function MC = monteCarlo(p, env, S)
rs = RandStream('mt19937ar', 'Seed', 11);
jumped = find([S.ev.executed]);
nJ = numel(jumped);
MC.idx = jumped;  MC.prob = zeros(1, nJ);  MC.n = p.mcN;
MC.margin = cell(1, nJ);  MC.phi = cell(1, nJ);
for q = 1:nJ
    i  = jumped(q);  pl = S.ev(i).plan;  ob = env.obs(i,:);
    zg = terrainZ(env, pl.xLaunch);
    xe = ob(1) + ob(2);
    ok = 0;  mg = NaN(1, p.mcN);  ph = NaN(1, p.mcN);
    for k = 1:p.mcN
        r  = randn(rs, 1, 3);
        v  = pl.v*(1 + p.mcSigV*r(1));
        th = pl.theta + p.mcSigTh*r(2);
        xl = pl.xLaunch + p.mcSigD*r(3);
        w0 = p.m*v*sin(th)*p.thrustOffset/p.I;
        fl = simulateFlight(p, zg + p.restZ + pl.comp, v*cos(th), v*sin(th), w0, true, zg, 0);
        if ~fl.hit, continue; end
        lc  = landingCheck(p, fl.vxTD, fl.vzTD, fl.wTD, fl.phiTD);
        mc  = minClearance(p, ob, xl + fl.x, fl.z, fl.phi, zg);
        rear = xl + fl.xTD - p.wb/2*cos(fl.phiTD) + p.hw*sin(fl.phiTD);
        mg(k) = rear - xe;
        ph(k) = wrapPi(fl.phiTD)*180/pi;
        if lc.stable && mc >= 0 && mg(k) >= 0
            ok = ok + 1;
        end
    end
    MC.prob(q) = ok/p.mcN;
    MC.margin{q} = mg;  MC.phi{q} = ph;
end
end

function CM = capabilityMap(p, env)
CM.h = [0.02 0.05 0.08 0.11 0.14 0.17 0.20];
CM.w = [0.10 0.20 0.30 0.40];
CM.v = NaN(numel(CM.h), numel(CM.w));
CM.mode = zeros(numel(CM.h), numel(CM.w));
env2 = env;
for a = 1:numel(CM.h)
    for b = 1:numel(CM.w)
        if CM.h(a) <= p.hRollMax
            CM.mode(a,b) = 1;
            continue;
        end
        env2.obs = [1.0, CM.w(b), CM.h(a)];
        pl = planJump(p, env2, 1, CM.h(a), CM.w(b), 0, 1);
        if pl.feasible
            CM.mode(a,b) = 2;  CM.v(a,b) = pl.v;
        else
            CM.mode(a,b) = 3;
        end
    end
end
end

%% =====================================================================
%%  EXTENSION 1 : AI-BASED OBSTACLE DECISION  (STEP / JUMP / AVOID)
%% =====================================================================
function [Er, Ej, Ea] = optionCosts(p, h, w, hw, soil, hl)
% Er = step-over energy (legs walk over), Ej = jump energy, Ea = detour energy (Inf = not possible)
Er = Inf;  Ej = Inf;  Ea = Inf;
if h <= p.hRollMax
    Er = p.m*p.g*h/p.etaDrive + 0.5*p.m*p.vCruise^2*h/p.r;
end
etaE = p.eta*soil*hl;
vmax = sqrt(max(etaE*p.k*p.maxTravel^2 - 2*p.m*p.g*p.maxTravel, 0)/p.m);
[TH, D1] = meshgrid((40:5:80)*pi/180, 0.15:0.05:0.45);
hc = h + p.safetyMargin;  hb = p.wb/2;
d2 = D1 + w;  xa = D1 - hb;  xb = d2 + hb;
ok = (xa.*tan(TH) - hc) > 0;
vA = p.g*xa.^2./(2*cos(TH).^2.*(xa.*tan(TH) - hc));
vB = p.g*xb.^2./(2*cos(TH).^2.*(xb.*tan(TH) - hc));
vR = p.g*(d2 + hb + p.landMargin)./sin(2*TH);
v  = sqrt(max(max(vA, vB), vR));
v(~ok) = Inf;
ve = springSpeed(p, v, TH);                 % running jump: spring only adds the missing velocity
vmin = min(ve(:));
if vmin <= vmax
    comp = (p.m*p.g + sqrt((p.m*p.g)^2 + etaE*p.k*p.m*vmin^2))/(etaE*p.k);
    Ej = 0.5*p.k*comp^2/p.etaWinch;
end
dy = hw + p.infl;
if dy <= p.laneLimit
    Ea = 0.05*p.m*p.g*(2*sqrt(5) - 4)*dy/p.etaDrive + 0.05;
end
end

function X = aiFeatures(M)
X = [M(:,1)/0.2, M(:,2)/0.4, M(:,3), (M(:,4) - 0.85)/0.15, (M(:,5) - 0.7)/0.3];
end

function [P, A1] = mlpForward(W1, b1, W2, b2, X)
A1 = tanh(bsxfun(@plus, X*W1, b1));
Z  = bsxfun(@plus, A1*W2, b2);
Z  = bsxfun(@minus, Z, max(Z, [], 2));
E  = exp(Z);
P  = bsxfun(@rdivide, E, sum(E, 2));
end

function net = trainDecisionNet(p)
rs = RandStream('mt19937ar', 'Seed', 21);
N  = 3000;
M  = [0.005 + 0.215*rand(rs, N, 1), 0.05 + 0.40*rand(rs, N, 1), 0.10 + 0.90*rand(rs, N, 1), ...
      0.85 + 0.15*rand(rs, N, 1),   0.70 + 0.30*rand(rs, N, 1)];
lab = zeros(N, 1);
for i = 1:N
    [Er, Ej, Ea] = optionCosts(p, M(i,1), M(i,2), M(i,3), M(i,4), M(i,5));
    e = [Er, Ej, Ea];
    if all(isinf(e)), lab(i) = 2; else, [~, lab(i)] = min(e); end
end
Mn = M + bsxfun(@times, randn(rs, N, 5), [0.003 0.01 0.01 0 0]);
X  = aiFeatures(Mn);
Y  = zeros(N, 3);  Y(sub2ind([N 3], (1:N)', lab)) = 1;
nTr = round(0.8*N);
Xt = X(1:nTr, :);      Yt = Y(1:nTr, :);
Xv = X(nTr+1:end, :);  lv = lab(nTr+1:end);

nH = 16;
prm = {0.5*randn(rs, 5, nH), zeros(1, nH), 0.5*randn(rs, nH, 3), zeros(1, 3)};
mo  = cellfun(@(a) zeros(size(a)), prm, 'UniformOutput', false);
vo  = mo;
lr = 0.03;  be1 = 0.9;  be2 = 0.999;  nEp = 1500;
loss = zeros(nEp, 1);  acc = zeros(nEp, 1);
for ep = 1:nEp
    [P, A1] = mlpForward(prm{1}, prm{2}, prm{3}, prm{4}, Xt);
    n   = size(Xt, 1);
    dZ2 = (P - Yt)/n;
    dZ1 = (dZ2*prm{3}').*(1 - A1.^2);
    g   = {Xt'*dZ1, sum(dZ1, 1), A1'*dZ2, sum(dZ2, 1)};
    for q = 1:4
        mo{q} = be1*mo{q} + (1 - be1)*g{q};
        vo{q} = be2*vo{q} + (1 - be2)*g{q}.^2;
        prm{q} = prm{q} - lr*(mo{q}/(1 - be1^ep))./(sqrt(vo{q}/(1 - be2^ep)) + 1e-8);
    end
    loss(ep) = -mean(log(sum(P.*Yt, 2) + 1e-12));
    Pv = mlpForward(prm{1}, prm{2}, prm{3}, prm{4}, Xv);
    [~, pr] = max(Pv, [], 2);
    acc(ep) = mean(pr == lv);
end
net.W1 = prm{1};  net.b1 = prm{2};  net.W2 = prm{3};  net.b2 = prm{4};
net.loss = loss;  net.acc = acc;  net.valAcc = acc(end);  net.nTrain = nTr;
end

function [c, prob] = aiDecide(net, h, w, hw, soil, hl)
X = aiFeatures([h w hw soil hl]);
prob = mlpForward(net.W1, net.b1, net.W2, net.b2, X);
[~, c] = max(prob);
end

function s = aiName(c)
nm = {'STEP', 'JUMP', 'AVOID'};
s = nm{c};
end

function s = fmtE(e)
if isnan(e) || isinf(e), s = 'n/a'; else, s = sprintf('%.2f J', e); end
end

%% =====================================================================
%%  EXTENSION 3 : PATH PLANNING
%% =====================================================================
function info = priorInfo(p, env)
n = size(env.obs, 1);
info.hw = env.obsHalfW(:);
info.crossE = zeros(n, 1);  info.jf = false(n, 1);
for i = 1:n
    zi = zoneAt(env, env.obs(i,1));
    [Er, Ej] = optionCosts(p, env.obs(i,3), env.obs(i,2), env.obsHalfW(i), env.zone.soft(zi), 1);
    info.crossE(i) = min(Er, Ej);
    info.jf(i)     = Ej < Er;
end
end

function R = planRoute(p, env, x0, y0, mode, forced, info)
dx = 0.1;  dy = 0.05;
yR = (-p.laneLimit:dy:p.laneLimit)';  nY = numel(yR);
xC = floor(x0/dx)*dx : dx : (env.xEnd + 0.5);  nC = numel(xC);
nO = size(env.obs, 1);
wRisk = 0;  if strcmp(mode, 'safest'), wRisk = 4; end
yG = laneGoal();
Lw = 3*(yG ~= 0)*(yR - yG).^2;

[XC, YR] = meshgrid(xC, yR);
Cx = zeros(nY, nC);
Rk = 0.5*(abs(YR)/p.laneLimit).^4;
for k = 1:nO
    xo = env.obs(k,1);  wo = env.obs(k,2);  hwk = info.hw(k);
    dxr = max(max(xo - XC, XC - (xo + wo)), 0);
    dyr = max(abs(YR) - hwk, 0);
    Rk  = Rk + exp(-hypot(dxr, dyr)/0.12);
    cols = find(xC >= xo - 0.15 & xC <= xo + wo + 0.15);
    if isempty(cols), continue; end
    if forced(k) == 1, hwk = 5; end
    rows = abs(yR) <= hwk + p.infl;
    if forced(k) == 2
        Cx(rows, cols) = Inf;
    else
        Cx(rows, cols) = info.crossE(k)/numel(cols);
        if info.jf(k)
            Rk(rows, cols) = Rk(rows, cols) + 2/numel(cols);
        end
    end
end

crrC = zeros(1, nC);
for c = 1:nC, crrC(c) = env.zone.crr(zoneAt(env, xC(c))); end
Wd = p.m*p.g/p.etaDrive;

J = Inf(nY, nC);  Pr = zeros(nY, nC);
j0 = min(max(round((y0 + p.laneLimit)/dy) + 1, 1), nY);
J(j0, 1) = 0;
for c = 2:nC
    for dj = -1:1
        ds = sqrt(dx^2 + (dj*dy)^2);
        for j = 1:nY
            jp = j - dj;
            if jp < 1 || jp > nY, continue; end
            if isinf(J(jp, c-1)), continue; end
            cost = J(jp, c-1) + Wd*crrC(c)*ds + 0.02*abs(dj) + Cx(j, c) + wRisk*Rk(j, c) + Lw(j);
            if cost < J(j, c), J(j, c) = cost;  Pr(j, c) = jp; end
        end
    end
end
Jt = J(:, nC) + 2*abs(yR - yG);
[best, jb] = min(Jt);
R.mode = mode;
if isinf(best)
    R.ok = false;  R.x = xC(:);  R.y = y0*ones(nC, 1);
    R.len = xC(end) - xC(1);  R.E = NaN;  R.risk = NaN;  R.cost = Inf;
    return;
end
rows = zeros(1, nC);  rows(nC) = jb;
for c = nC:-1:2, rows(c-1) = Pr(rows(c), c); end
R.ok = true;  R.x = xC(:);  R.y = yR(rows);
R.len  = sum(sqrt(dx^2 + diff(R.y).^2));
idx    = sub2ind([nY nC], rows, 1:nC);
cx     = Cx(idx);
R.E    = sum(Wd*crrC(2:end).*sqrt(dx^2 + diff(R.y).^2)') + sum(cx(isfinite(cx)));
R.risk = sum(Rk(idx));
R.cost = best;
end

function y = routeY(route, x)
x = min(max(x, route.x(1)), route.x(end));
y = interp1(route.x, route.y, x, 'linear');
end

%% =====================================================================
%%  EXTENSION 5 : FAULT INJECTION
%% =====================================================================
function fs = faultState(env, x)
fs = struct('mask', 0, 'lNoise', 1, 'lRange', 1, 'lSlow', 1, 'mScale', 1, 'mEff', 1, 'sLoss', 0, 'iSlow', 1);
if ~env.faultsOn, return; end
for i = 1:numel(env.faults)
    f = env.faults(i);
    if x >= f.x0 && x < f.x1
        switch f.type
            case 'LIDAR'
                fs.mask = bitor(fs.mask, 1);  fs.lNoise = f.a;  fs.lRange = f.b;  fs.lSlow = 0.6;
            case 'MOTOR'
                fs.mask = bitor(fs.mask, 2);  fs.mScale = f.a;  fs.mEff = f.b;
            case 'SPRING'
                fs.mask = bitor(fs.mask, 4);  fs.sLoss = f.a;
            case 'IMU'
                fs.mask = bitor(fs.mask, 8);  fs.iSlow = f.a;
        end
    end
end
end

function S = faultMonitor(S, p, env) %#ok<INUSL>
fs = faultState(env, S.x);
if fs.mask ~= S.fm
    ch = bitxor(fs.mask, S.fm);
    for b = [1 2 4 8]
        if bitand(ch, b) == 0, continue; end
        S = say(S, faultText(b, bitand(fs.mask, b) > 0, fs));
    end
    S.fm = fs.mask;
end
S.lNoise = fs.lNoise;  S.lRange = fs.lRange;  S.mScale = fs.mScale;  S.mEff = fs.mEff;
S.sLoss  = fs.sLoss;   S.vScale = min(fs.lSlow, fs.iSlow);
end

function ln = faultText(b, on, fs)
switch b
    case 1
        if on
            ln = {'FAULT DETECTED: LiDAR DEGRADED', ...
                  sprintf('noise x%.0f, range -%.0f %%, speed limit %.0f %%', fs.lNoise, 100*(1 - fs.lRange), 100*fs.lSlow), ...
                  'planning with inflated uncertainty'};
        else
            ln = {'FAULT CLEARED: LiDAR nominal'};
        end
    case 2
        if on
            ln = {'FAULT DETECTED: LEG ACTUATORS DEGRADED', ...
                  sprintf('gait speed x%.2f, efficiency x%.2f', fs.mScale, fs.mEff), 'battery drain rises'};
        else
            ln = {'FAULT CLEARED: leg actuators nominal'};
        end
    case 4
        if on
            ln = {'FAULT DETECTED: LEG COIL SPRING DEGRADED', ...
                  sprintf('usable spring energy -%.0f %%', 100*fs.sLoss), 're-planning with degraded spring model'};
        else
            ln = {'FAULT CLEARED: leg coil spring nominal'};
        end
    otherwise
        if on
            ln = {'FAULT DETECTED: IMU GYRO BIAS', 'attitude estimate unreliable', ...
                  sprintf('safe mode: speed limited to %.0f %%', 100*fs.iSlow)};
        else
            ln = {'FAULT CLEARED: IMU nominal'};
        end
end
end

%% =====================================================================
%%  EXTENSION 2 : DYNAMIC MOVING OBSTACLES
%% =====================================================================
function y = dynY(d, t0, t)
if isnan(t0) || t < t0, y = d.y0; return; end
y  = d.y0 + d.vy*(t - t0);
lo = min(d.y0, -d.y0);  hi = max(d.y0, -d.y0);
y  = min(max(y, lo), hi);
end

function [S, vLim, yielding] = dynamicStep(S, p, env)
vLim = Inf;  yielding = false;
if ~env.dynOn, return; end
for i = 1:numel(env.dyn)
    d = env.dyn(i);  D = S.dyn(i);
    if D.state == 3, continue; end
    if D.state == 0
        if S.x < d.trigX, continue; end
        D.state = 1;  D.t0 = S.t;
        if ~isnan(S.dynT0(i)), D.t0 = S.dynT0(i) - S.tOff; end
        S.dyn(i) = D;
        if d.rock
            kind = 'ROCK';
        else
            kind = 'object';
        end
        S = say(S, {sprintf('DYNAMIC: moving %s #%d detected', kind, i), ...
                    sprintf('%s at x = %.1f m, radar tracking started', d.name, d.x), ...
                    'estimating velocity (least-squares fit)'});
        D = S.dyn(i);
    end
    gap  = (d.x - d.len/2) - (S.x + p.L/2);
    back = (d.x + d.len/2) - (S.x - p.L/2);
    if back < -0.05
        D.state = 3;  S.dyn(i) = D;
        S = say(S, {sprintf('DYNAMIC: object #%d passed', i), 'tracking released'});
        continue;
    end
    yT = dynY(d, D.t0, S.t);
    ym = yT + p.dynNoise*S.lNoise*randn;
    D.buf = [D.buf; S.t, ym];
    if size(D.buf, 1) > 25, D.buf(1,:) = []; end
    if size(D.buf, 1) >= 8
        c = polyfit(D.buf(:,1) - S.t, D.buf(:,2), 1);
        yh = c(2);  vh = c(1);
    else
        yh = ym;  vh = 0;
    end
    tArr = max(gap, 0)/0.5;  tPass = (d.len + p.L)/0.5;
    s = linspace(tArr, tArr + tPass, 7);
    yR = routeY(S.route, d.x);
    conflict = any(abs(yh + vh*s - yR) < d.hw + p.robotHW + 0.10);
    wasYield = (D.state == 2);
    msg = {};
    if conflict
        D.state = 2;
        if ~wasYield
            msg = {sprintf('YIELD: conflict with object #%d predicted', i), ...
                   sprintf('est. v = %+.2f m/s, y = %+.2f m, gap %.2f m', vh, yh, gap), 'braking to a controlled stop'};
        end
        distStop = gap - 0.40;
        if distStop < 0.003, vLim = 0; else, vLim = min(vLim, sqrt(2*1.5*distStop)); end
        yielding = true;
    elseif wasYield
        D.state = 1;
        msg = {'CLEAR: lane is free', sprintf('object #%d moved away - resuming', i)};
    end
    D.log(end+1, :) = [S.t, yT, yh, vh, D.state]; %#ok<AGROW>
    S.dyn(i) = D;
    if ~isempty(msg), S = say(S, msg); end
end
end

%% =====================================================================
%%  EXTENSION 4 : SENSOR LOG
%% =====================================================================
function D = buildSensorLog(p, env, S)
F = S.F;  N = size(F, 1);
t = F(:,1);  x = F(:,2);  phi = F(:,4);  mode = F(:,8);  vz = F(:,14);  y = F(:,17);
rs = RandStream('mt19937ar', 'Seed', 33);
lid = NaN(N, 1);  sLoss = zeros(N, 1);
for k = 1:N
    fs = faultState(env, x(k));
    sLoss(k) = fs.sLoss;
    if mode(k) == 7 || mode(k) == 8, continue; end
    xs = x(k) + p.L/2 + 0.03;
    Rg = p.senseRange*fs.lRange;
    r  = Rg;
    for i = 1:size(env.obs, 1)
        xo = env.obs(i,1);
        if xo > xs && (xo - xs) <= Rg && abs(y(k)) < env.obsHalfW(i) + 0.10
            r = min(r, xo - xs);
        end
    end
    if env.dynOn
        for i = 1:numel(env.dyn)
            d = env.dyn(i);  yd = dynY(d, S.dyn(i).t0, t(k));  xd = d.x - d.len/2;
            if xd > xs && (xd - xs) <= Rg && abs(yd - y(k)) < d.hw + 0.10
                r = min(r, xd - xs);
            end
        end
    end
    r = min(max(r + p.senseNoise*fs.lNoise*randn(rs), 0), Rg);
    if fs.lNoise > 1 && rand(rs) < 0.12, r = NaN; end
    lid(k) = r;
end

dtv = max(diff(t), 1e-3);
imu = bitand(F(:,19), 8) > 0;
om  = [diff(phi)./dtv; 0]*180/pi;
bias = zeros(N, 1);  nz = 0.3*ones(N, 1);
bias(imu) = 6 + 2*sin(2*t(imu));  nz(imu) = 2;
gyro = min(max(om + bias + nz.*randn(rs, N, 1), -600), 600);

az = [diff(vz)./dtv; 0];
az = (az + p.g)/p.g;
az = conv(az, ones(5,1)/5, 'same');
az = min(max(az, -1), 6) + (0.02 + 0.10*imu).*randn(rs, N, 1);

D = [lid, gyro, az, 100*batterySOC(p, t, F(:,10)), F(:,11), 100*F(:,16).*(1 - sLoss), F(:,12)];
end

%% =====================================================================
%%  SIMULATION
%% =====================================================================
function S = runSimulation(p, env, net, rb)
if nargin < 4
    rb = struct('id', 1, 'y0', 0, 'yGoal', 0, 'nScan', p.nScan, 'quiet', false, 'tOff', 0, ...
                'dynT0', NaN(1, numel(env.dyn)));
end
p.nScan = rb.nScan;  laneGoal(rb.yGoal);
nObs = size(env.obs,1);
evT = struct('idx',0,'hM',NaN,'wM',NaN,'distM',NaN,'decision','-','Eroll',NaN,'Ejump',NaN, ...
             'plan',[],'executed',false,'tSense',NaN,'tDone',NaN,'landPhi',NaN,'clear',NaN, ...
             'arcX',[],'arcZ',[],'sigH',NaN, ...
             'Eavoid',NaN,'cAI',0,'pAI',NaN,'dPhys',0,'override',false,'yL',0,'hwM',NaN);
S.F = zeros(0,19);   S.t = 0;  S.x = 0.35;  S.phi = 0;  S.wheel = 0;  S.E = 0;
S.y = rb.y0;  S.yaw = 0;  S.fm = 0;
S.vNow = 0;                                   % actual forward speed of the last walking step (m/s)
S.quiet = rb.quiet;  S.tOff = rb.tOff;  S.dynT0 = rb.dynT0;
S.lNoise = 1;  S.lRange = 1;  S.mScale = 1;  S.mEff = 1;  S.sLoss = 0;  S.vScale = 1;
S.z = rollPose(p, env, S.x);
S.health = 1;  S.jumps = 0;
S.msgs = {};  S.msgId = 0;  S.active = 0;  S.status = 'RUNNING';
S.ev = repmat(evT, nObs, 1);
S.kick = struct('happened',false,'tumble',[],'sr',[],'lcSR',[],'tStart',NaN,'tEnd',NaN);
S.net = net;
S.dyn = repmat(struct('state',0,'t0',NaN,'buf',zeros(0,2),'log',zeros(0,5)), numel(env.dyn), 1);

S.info   = priorInfo(p, env);
if isfield(rb, 'info'), S.info = rb.info; end
S.forced = zeros(nObs, 1);
Rs = planRoute(p, env, S.x, S.y, 'shortest', S.forced, S.info);
Ra = planRoute(p, env, S.x, S.y, 'safest',   S.forced, S.info);
S.R0 = struct('shortest', Rs, 'safest', Ra);
if strcmp(env.routeMode, 'shortest'), S.route = Rs; else, S.route = Ra; end
S.routeT = 0;  S.routes = {S.route};

S = say(S, {'SYSTEM READY', 'Digital twin V7 - spring-legged RUNNING jumper, AI decision, route planner, radar, fault monitor online'});
S = say(S, {sprintf('ROUTE PLANNED: %s', upper(env.routeMode)), ...
            sprintf('shortest %.2f m / %.1f J / risk %.1f', Rs.len, Rs.E, Rs.risk), ...
            sprintf('safest   %.2f m / %.1f J / risk %.1f', Ra.len, Ra.E, Ra.risk)});
S = addRow(S, 1, 0, 0, 0, 0, 0, 0);

sensed = false(nObs,1);
jobIdx = 0;  xTrig = NaN;  kickDone = (rb.id > 1);
guard = 0;

while S.x < env.xEnd && guard < 100000 && ~strcmp(S.status,'BLOCKED')
    guard = guard + 1;
    S = faultMonitor(S, p, env);

    if jobIdx == 0
        i = find(~sensed, 1);
        if ~isempty(i) && (env.obs(i,1) - (S.x + p.L/2)) <= p.senseRange*S.lRange
            sensed(i) = true;
            [S, jobIdx, xTrig] = senseAndDecide(S, p, env, i);
            if strcmp(S.status,'BLOCKED'), break; end
        end
    end

    if ~kickDone && jobIdx == 0 && S.x >= env.kick.x
        kickDone = true;
        S = kickEvent(S, p, env);
        continue;
    end

    if jobIdx > 0
        % ---- running jump: keep walking at the run-up speed until the trigger point, never stop ----
        if S.x >= xTrig - 1e-9
            zg = terrainZ(env, S.x);
            ji = jobIdx;
            S = executeJump(S, p, env, ji, S.ev(ji).plan, zg);
            jobIdx = 0;
            S.health = S.health*(1 - p.fatigue);
            S.jumps  = S.jumps + 1;
            left = max(0, floor(log(p.healthMin)/log(1 - p.fatigue)) - S.jumps);
            S = say(S, {sprintf('HEALTH: leg spring %.1f %% after jump #%d', 100*S.health, S.jumps), ...
                        sprintf('predicted jumps until service: %d', left)});
            continue;
        end
        v = p.vJump;
    else
        v = p.vCruise;
    end

    [S, vDyn, yielding] = dynamicStep(S, p, env);
    v = min(v*S.vScale, vDyn);
    mode = 1;
    if yielding
        mode = 11;
    elseif abs(S.y) > 0.03 || abs(routeY(S.route, S.x) - S.y) > 0.03
        mode = 10;
    end
    S = rollStep(S, p, env, v, mode);

    for q = 1:nObs
        if (strcmp(S.ev(q).decision,'STEP') || strcmp(S.ev(q).decision,'AVOID')) && isnan(S.ev(q).tDone) && ...
           (S.x - p.wb/2) > (env.obs(q,1) + env.obs(q,2) + 0.02)
            S.ev(q).tDone = S.t;
        end
    end
end

if ~strcmp(S.status,'BLOCKED'), S.status = 'COMPLETE'; end
S = say(S, {sprintf('MISSION %s', S.status), sprintf('distance %.1f m, energy used %.1f J', S.x, S.E)});
for j = 1:60
    S = rollStep(S, p, env, 0, 9);
end
end

function [S, jobIdx, xTrig] = senseAndDecide(S, p, env, i)
jobIdx = 0;  xTrig = NaN;
xo   = env.obs(i,1);
dist = xo - (S.x + p.L/2);
Rn   = p.senseNoise*S.lNoise;

[hM, sh] = fuseScans(env.obs(i,3), Rn, p.nScan);
[wM, sw] = fuseScans(env.obs(i,2), Rn, p.nScan);
[yM, sy] = fuseScans(env.obsHalfW(i), Rn, p.nScan);
hM = max(0.005, hM);  wM = max(0.05, wM);  yM = max(0.05, yM);
hPlan = hM + p.kSigma*sh;
wPlan = wM + p.kSigma*sw;
yPlan = yM + p.kSigma*sy;

zi   = zoneAt(env, xo);
soil = env.zone.soft(zi);
zg   = terrainZ(env, xo);
hEff = S.health*(1 - S.sLoss);

S = say(S, {sprintf('SENSE: obstacle #%d (%s) at %.2f m', i, env.obsName{i}, dist), ...
            sprintf('h = %.1f +/- %.2f mm, w = %.0f mm, half-width %.0f mm  [%s]', 1000*hM, 1000*sh, 1000*wM, 1000*yM, env.zone.name{zi}), ...
            sprintf('%d scans fused (Kalman), plan with h+%d sigma%s', p.nScan, p.kSigma, ternary(S.lNoise > 1, '  [LiDAR FAULT]', ''))});

rollOK = hPlan <= p.hRollMax;
Eroll  = Inf;
if rollOK
    Eroll = p.m*p.g*hM/p.etaDrive + 0.5*p.m*p.vCruise^2*hM/p.r;
end
pp = p;  pp.eta = p.eta*hEff;
plan  = planJump(pp, env, i, hPlan, wPlan, zg, soil);
if env.asOn && plan.feasible
    c0 = plan.comp;
    plan = adaptSpring(pp, env, i, plan, hPlan, wPlan, zg, soil, S.lNoise);
    S = say(S, {sprintf('ADAPTIVE SPRING: crouch %.1f -> %.1f mm (x%.2f)', 1000*c0, 1000*plan.comp, plan.asMul), ...
                sprintf('worst-case landing margin %.0f mm over 5 sigma cases', 1000*plan.asMargin), ...
                sprintf('uncertainty x%.1f, soil %.0f%%, spring health %.0f%%', S.lNoise, 100*soil, 100*hEff)});
end
Ejump = Inf;
if plan.feasible, Ejump = plan.Ecost; end

[~, ~, EaApprox] = optionCosts(p, hM, wM, yPlan, soil, hEff);
S.info.hw(i) = yPlan;  S.info.crossE(i) = min(Eroll, Ejump);  S.info.jf(i) = Ejump < Eroll;
Eavoid = Inf;  Ravoid = [];
if isfinite(EaApprox)
    fz = S.forced;  fz(i) = 2;
    Ravoid = planRoute(p, env, S.x, S.y, env.routeMode, fz, S.info);
    if Ravoid.ok, Eavoid = EaApprox; end
end

opt = [Eroll, Ejump, Eavoid];
S.ev(i).idx = i;  S.ev(i).hM = hM;  S.ev(i).wM = wM;  S.ev(i).hwM = yM;  S.ev(i).distM = dist;
S.ev(i).Eroll = Eroll;  S.ev(i).tSense = S.t;  S.ev(i).sigH = sh;
S.ev(i).Ejump  = ternary(isfinite(Ejump), Ejump, NaN);
S.ev(i).Eavoid = ternary(isfinite(Eavoid), Eavoid, NaN);

if all(isinf(opt))
    S.ev(i).decision = 'BLOCKED';
    S = say(S, {'DECIDE: NO SAFE OPTION FOUND', 'cannot step over, jump or walk around - mission halted'});
    S.status = 'BLOCKED';
    return;
end

[~, dPhys] = min(opt);
dFinal = dPhys;  cAI = 0;  pAI = NaN;  override = false;
if env.aiOn
    [cAI, prob] = aiDecide(S.net, hM, wM, yM, soil, hEff);
    pAI = prob(cAI);
    if isfinite(opt(cAI)) && pAI >= p.aiConf
        dFinal = cAI;
    else
        override = true;
    end
    S = say(S, {sprintf('AI: predicts %s (%.0f %% confidence)', aiName(cAI), 100*pAI), ...
                sprintf('P(step)=%.2f  P(jump)=%.2f  P(avoid)=%.2f', prob(1), prob(2), prob(3)), ...
                sprintf('physics check: step %s | jump %s | avoid %s', fmtE(Eroll), fmtE(Ejump), fmtE(Eavoid))});
    if override
        S = say(S, {'SUPERVISOR: AI proposal REJECTED', sprintf('infeasible or low confidence - using physics: %s', aiName(dPhys))});
    elseif cAI ~= dPhys
        S = say(S, {'SUPERVISOR: AI proposal accepted', sprintf('feasible, differs from min-energy %s', aiName(dPhys))});
    else
        S = say(S, {'SUPERVISOR: AI proposal verified', 'matches the minimum-energy physics choice'});
    end
end
S.ev(i).cAI = cAI;  S.ev(i).pAI = pAI;  S.ev(i).dPhys = dPhys;  S.ev(i).override = override;

if dFinal == 3
    S.forced(i) = 2;
    S.route = Ravoid;
else
    S.forced(i) = 1;
    Rc = planRoute(p, env, S.x, S.y, env.routeMode, S.forced, S.info);
    if Rc.ok, S.route = Rc; end
end
S.routeT(end+1) = S.t;  S.routes{end+1} = S.route;

switch dFinal
    case 3
        S.ev(i).decision = 'AVOID';
        S = say(S, {'DECIDE: walk AROUND the obstacle', ...
                    sprintf('route re-planned: %.2f m remaining, detour energy %.2f J', S.route.len, Eavoid)});
    case 1
        S.ev(i).decision = 'STEP';
        S = say(S, {'DECIDE: keep WALKING, step over (minimum energy)', sprintf('E_step = %.2f J', Eroll)});
    otherwise
        S.ev(i).decision = 'JUMP';  S.ev(i).plan = plan;
        S.ev(i).arcX = plan.arcX;   S.ev(i).arcZ = plan.arcZ;
        if rollOK
            why = sprintf('E_jump %.2f J chosen over E_step %.2f J', Ejump, Eroll);
        else
            why = sprintf('h > leg step-over limit %.0f mm, no way around', 1000*p.hRollMax);
        end
        S = say(S, {'DECIDE: JUMP over obstacle ON THE MOVE (all four spring legs)', why});
        S = say(S, {sprintf('PLAN: angle %.0f deg, launch %.2f m/s', plan.theta*180/pi, plan.v), ...
                    sprintf('run-up %.2f m/s kept (no stop): spring adds dvx %.2f, dvz %.2f m/s', plan.vApp, plan.vx - plan.vApp, plan.vz), ...
                    sprintf('leg crouch %.1f mm, F = %.0f N, standoff %.0f mm', 1000*plan.comp, plan.force, 1000*plan.d1), ...
                    sprintf('spring energy %.2f J (ground %.0f%%, health %.0f%%)', plan.Espring, 100*soil, 100*hEff)});
        S = say(S, {sprintf('CHECK: landing %s', ternary(plan.lc.stable,'STABLE','UNSTABLE')), ...
                    sprintf('pitch %.1f deg (uncontrolled %.0f deg)', plan.phiC, plan.phiU), ...
                    sprintf('tip margin %.0f%%, impact vz %.2f m/s', 100*plan.lc.tipMargin, plan.fl.vzTD), ...
                    sprintf('rejected plans: %d unsafe of %d', plan.rejStab + plan.rejClear + plan.rejFlat, plan.nCand)});
        % trigger point: start of the moving crouch, so that the release happens exactly at plan.xLaunch
        tpP  = 2*plan.comp/plan.v;
        xTrig = plan.xLaunch - launchTravel(plan.vApp, plan.vx, p.nCrouchRun, tpP);
        jobIdx = i;  S.active = i;
        % NOTE: the robot does NOT stop here - the main loop keeps it walking at p.vJump up to xTrig.
end
end

function S = executeJump(S, p, env, i, plan, zg)
S.ev(i).yL = S.y;
vApp = min(max(S.vNow, 0), plan.vx);          % actual forward speed at the trigger point
tp   = 2*plan.comp/plan.v;
if vApp > 0.05
    % stretch / shorten the crouch so the release point stays exactly at plan.xLaunch
    extT = launchTravel(vApp, plan.vx, 0, tp);
    nC   = round((plan.xLaunch - S.x - extT)/(vApp*0.01));
    nC   = min(max(nC, 5), 200);
else
    vApp = 0;  nC = 50;                       % robot was held up by traffic: standing jump
end
txt.charge  = {sprintf('ACTION: legs crouching ON THE MOVE (run-up %.2f m/s)', vApp), ...
               sprintf('target %.1f mm  (F = %.0f N)', 1000*plan.comp, plan.force)};
txt.release = {'ACTION: RELEASE - springs extend the legs', ...
               sprintf('launch %.2f m/s at %.0f deg (dvx %.2f, dvz %.2f from springs)', plan.v, plan.theta*180/pi, plan.vx - vApp, plan.vz)};
txt.flight  = {'FLIGHT: spring legs retracted, reaction-wheel control ON', ...
               sprintf('air time %.2f s, peak torque %.2f Nm', plan.fl.T, plan.fl.tauPeak), ...
               sprintf('wheel absorbs L = %.3f kg m2/s', abs(plan.fl.Hrw))};
S = launchSequence(S, p, env, [3 4 5], plan.comp, tp, plan.vx, plan.vz, plan.fl, txt, vApp, nC);

fl = plan.fl;
lc = landingCheck(p, fl.vxTD, fl.vzTD, fl.wTD, fl.phiTD);
mc = minClearance(p, env.obs(i,:), S.ev(i).arcX, fl.z, fl.phi, zg);
S = say(S, {'LANDING detected (foot contact sensors)', ...
            sprintf('pitch %.1f deg, impact vz %.2f m/s', wrapPi(fl.phiTD)*180/pi, fl.vzTD), ...
            sprintf('tip margin %.0f%% -> %s', 100*lc.tipMargin, ternary(lc.stable,'STABLE','UNSTABLE')), ...
            sprintf('min clearance over obstacle %.0f mm', 1000*mc)});
S = settle(S, p, env, fl.vxTD, 40);
S.ev(i).executed = true;  S.ev(i).tDone = S.t;
S.ev(i).landPhi = wrapPi(fl.phiTD)*180/pi;  S.ev(i).clear = mc;
S.active = 0;
end

function S = launchSequence(S, p, env, modes, comp, tp, vxL, vzL, fl, txt, vApp, nC)
% vApp > 0 : running jump - x advances during crouch and leg extension, the robot never stands still.
% vApp = 0 : standing launch (used for the self-righting hop).
moving = vApp > 0;
envY = envAtY(env, S.y);
S = say(S, txt.charge);
E0 = S.E;  Ej = 0.5*p.k*comp^2/p.etaWinch;
for j = 1:nC
    a = j/nC;  a = a*a*(3 - 2*a);
    c = comp*a;
    Enew = E0 + Ej*a^2;
    S.t = S.t + 0.01;
    if moving
        S.x = S.x + vApp*0.01;
        [S.z, S.phi] = rollPose(p, envY, S.x);
        S.wheel = S.wheel + vApp*0.01/p.r;
    end
    pw = (Enew - S.E)/0.01;  S.E = Enew;
    S = addRow(S, modes(1), c, 0, pw, p.k*c, vApp, 0);
end
S = say(S, txt.release);
z0 = S.z;  nT = 6;  dtE = tp/nT;
for j = 1:nT
    a = j/nT;
    vxj = vApp + (vxL - vApp)*a;              % springs accelerate the robot from the run-up speed to the launch speed
    S.t = S.t + dtE;
    if moving
        S.x = S.x + vxj*dtE;
    end
    S.z = z0 + comp*a^2;
    c   = comp*(1 - a^2);
    if moving, vShow = vxj; else, vShow = vxL*a; end
    S = addRow(S, modes(2), c, comp*a^2, 0, p.k*c, vShow, vzL*a);
end
S = say(S, txt.flight);
x0 = S.x;  t0 = S.t;
idx = unique([5:4:fl.n, fl.n]);
for j = 1:numel(idx)
    q = idx(j);  tt = fl.t(q);
    S.t = t0 + tt;  S.x = x0 + fl.x(q);  S.z = fl.z(q);  S.phi = fl.phi(q);
    u = min(tt/0.25, 1);  u = u*u*(3 - 2*u);
    S = addRow(S, modes(3), 0, comp*(1 - u), 0, 0, fl.vx, fl.vz(q));
end
end

function S = kickEvent(S, p, env)
S.yaw = 0;
zg   = terrainZ(env, S.x);
soil = env.zone.soft(zoneAt(env, S.x))*(1 - S.sLoss);
w0   = env.kick.omega;
r0x  = sign(w0)*p.wb/2;
r0   = [r0x, p.restZ];
xP   = S.x - r0x;
tm   = tumbleSim(p, r0, w0);
S.kick.happened = true;  S.kick.tumble = tm;  S.kick.tStart = S.t;

S = say(S, {'EVENT: external side impact!', sprintf('angular rate imposed: %.1f rad/s', w0), ...
            sprintf('rotation energy %.2f J vs tip barrier %.2f J', tm.Ek, tm.Ebar)});
n = numel(tm.t);  idx = unique([1:10:n, n]);  t0 = S.t;
for j = 1:numel(idx)
    q = idx(j);  d = tm.dphi(q);
    rx = cos(d)*r0(1) - sin(d)*r0(2);
    rz = sin(d)*r0(1) + cos(d)*r0(2);
    [~, dzl] = lowestPoint(p, d);
    S.t = t0 + tm.t(q);  S.x = xP + rx;  S.phi = d;
    S.z = max(zg + rz, zg - dzl);
    S = addRow(S, 7, 0, 0, 0, 0, 0, 0);
end
if ~tm.inverted
    S = say(S, {'Impact absorbed - robot rocked back', 'IMU: upright, no recovery needed'});
    S = settle(S, p, env, 0, 30);
    S.kick.tEnd = S.t;
    return;
end

S = say(S, {'TUMBLE: robot is UPSIDE-DOWN', sprintf('IMU pitch = %.0f deg -> start self-righting', S.phi*180/pi)});
for j = 1:40
    S.t = S.t + 0.01;
    S = addRow(S, 7, 0, 0, 0, 0, 0, 0);
end

sr = planSelfRight(p, soil, zg);
S.kick.sr = sr;
S = say(S, {'SELF-RIGHT PLAN: spring + dorsal flip arm', ...
            sprintf('lever arm %.0f mm, hop %.2f m/s', 1000*p.armSR, sr.vz), ...
            sprintf('angular momentum L = I*w = %.3f kg m2/s', sr.L), ...
            sprintf('spin %.1f rad/s, spring %.1f mm, F = %.0f N', sr.w, 1000*sr.comp, sr.force)});
txt.charge  = {'SELF-RIGHT: charging flip spring', sprintf('target %.1f mm', 1000*sr.comp)};
txt.release = {'SELF-RIGHT: RELEASE - flip arm kick', 'hop + 180 deg rotation in the air'};
txt.flight  = {'FLIGHT: rotating to feet-down', sprintf('air time %.2f s', sr.fl.T)};
S = launchSequence(S, p, env, [8 8 8], sr.comp, sr.tp, 0, sr.vz, sr.fl, txt, 0, 50);

fl = sr.fl;
lc = landingCheck(p, 0, fl.vzTD, fl.wTD, fl.phiTD);
S.kick.lcSR = lc;
S = say(S, {'LANDING detected after self-right hop', ...
            sprintf('final pitch %.1f deg (limit %.0f deg)', wrapPi(fl.phiTD)*180/pi, p.phiLimit*180/pi), ...
            sprintf('tip margin %.0f%% -> %s', 100*lc.tipMargin, ternary(lc.stable,'STABLE','UNSTABLE'))});
S = settle(S, p, env, 0, 40);
if lc.stable
    S = say(S, {'RECOVERY SUCCESS - standing on all four spring legs', 'resuming spring-leg gait'});
else
    S = say(S, {'RECOVERY WARNING - landing unstable', 'resuming with caution'});
end
S.kick.tEnd = S.t;
end

function S = rollStep(S, p, env, v, mode)
% one 10 ms walking step.  S.wheel accumulates distance/p.r (gait phase source).
dt  = 0.01;
v   = v*S.mScale;
S.vNow = v;                                   % remembered for the running jump
dxs = v*dt;
xNew = S.x + dxs;
yT  = routeY(S.route, S.x);
dyS = min(max(yT - S.y, -0.5*dxs), 0.5*dxs);
yNew = S.y + dyS;
if dxs > 1e-6
    S.yaw = 0.85*S.yaw + 0.15*atan2(dyS, dxs);
end
[zNew, phiNew] = rollPose(p, envAtY(env, yNew), xNew);
dz = (zNew - S.z)/dt;
zi = zoneAt(env, S.x);
ds = sqrt(dxs^2 + dyS^2);
P  = max(0, p.m*p.g*dz + env.zone.crr(zi)*p.m*p.g*ds/dt)/(p.etaDrive*S.mEff);
S.E = S.E + P*dt;
S.t = S.t + dt;  S.x = xNew;  S.y = yNew;  S.z = zNew;  S.phi = phiNew;
S.wheel = S.wheel + ds/p.r;
S = addRow(S, mode, 0, 0, P, 0, v, dz);
end

function S = settle(S, p, env, vxTD, nFrames)
dt = 0.01;  tauS = 0.06;  tauB = 0.08;
x0 = S.x;  z0 = S.z;  phi0 = S.phi;
[zt0, pt0] = rollPose(p, env, x0);
e0 = max(z0 - zt0, 0);
ep = phi0 - pt0;
for j = 1:nFrames
    tt   = j*dt;
    xNew = x0 + vxTD*tauS*(1 - exp(-tt/tauS));
    [zt, pt] = rollPose(p, env, xNew);
    b    = exp(-tt/tauB);
    zNew = zt + e0*b + 0.008*b*abs(sin(25*tt));
    phiN = pt + ep*b;
    [~, dzl] = lowestPoint(p, phiN);
    zNew = max(zNew, terrainZ(env, xNew) - dzl);
    S.wheel = S.wheel + (xNew - S.x)/p.r;
    S.t = S.t + dt;  S.x = xNew;  S.z = zNew;  S.phi = phiN;
    S = addRow(S, 6, 0, 0, 0, 0, vxTD*exp(-tt/tauS), 0);
end
end

function S = addRow(S, mode, comp, leg, pw, fsp, vx, vz)
S.F(end+1,:) = [S.t, S.x, S.z, S.phi, S.wheel, comp, leg, mode, S.msgId, S.E, pw, fsp, vx, vz, S.active, S.health, ...
                S.y, S.yaw, S.fm];
end

function S = say(S, lines)
if ischar(lines), lines = {lines}; end
S.msgs{end+1} = lines;
S.msgId = numel(S.msgs);
if ~S.quiet
    fprintf('[t=%6.2f s | x=%5.2f m] %s\n', S.t, S.x, strjoin(lines, ' | '));
end
end

function out = ternary(cond, a, b)
if cond, out = a; else, out = b; end
end

%% =====================================================================
%%  EXTENSION 6 : ADAPTIVE SPRING CONTROL
%% =====================================================================
function plan = adaptSpring(pp, env, iObs, plan, hM, wM, zg, soil, lNoise)
if ~plan.feasible, return; end
etaE = pp.eta*soil;  hb = pp.wb/2;
ob = [env.obs(iObs,1), wM, hM];  xe = ob(1) + ob(2);  xl = plan.xLaunch;
ks  = pp.kSigma*sqrt(max(lNoise, 1));
dv  = ks*pp.mcSigV*[0 -1 1 0 0];
dth = ks*pp.mcSigTh*[0 0 0 -1 1];
want = 0.03;
comp0 = plan.comp;
best = struct('cm', 1, 'worst', -Inf, 'comp', comp0, 'v', plan.v, 'fl', []);
for cm = 1:0.02:1.30
    comp = comp0*cm;
    if comp > pp.maxTravel, break; end
    ve = sqrt(max((etaE*pp.k*comp^2 - 2*pp.m*pp.g*comp)/pp.m, 0));   % velocity delivered by the springs
    v  = launchFromSpring(pp, ve, plan.theta);                       % world launch speed (run-up + spring)
    worst = Inf;  flNom = [];
    for q = 1:numel(dv)
        vq = v*(1 + dv(q));  thq = plan.theta + dth(q);
        w0 = pp.m*vq*sin(thq)*pp.thrustOffset/pp.I;
        fl = simulateFlight(pp, zg + pp.restZ + comp, vq*cos(thq), vq*sin(thq), w0, true, zg, 0);
        mg = -1;
        if fl.hit
            lc   = landingCheck(pp, fl.vxTD, fl.vzTD, fl.wTD, fl.phiTD);
            mc   = minClearance(pp, ob, xl + fl.x, fl.z, fl.phi, zg);
            rear = xl + fl.xTD - hb*cos(fl.phiTD) + pp.hw*sin(fl.phiTD);
            flat = isFlat(env, xl - pp.L/2, xl + fl.xTD + hb + 0.2, zg, iObs);
            if lc.stable && mc >= 0.01 && flat, mg = rear - xe; end
        end
        if q == 1
            flNom = fl;
            if mg < 0
                worst = -2;  break;
            end
        end
        worst = min(worst, mg);
    end
    if worst > -1.5 && worst > best.worst + 1e-9
        best = struct('cm', cm, 'worst', worst, 'comp', comp, 'v', v, 'fl', flNom);
    end
    if worst >= want, break; end
end
if isempty(best.fl), return; end
fl = best.fl;  v = best.v;  th = plan.theta;  comp = best.comp;
plan.comp0 = comp0;  plan.asMul = best.cm;  plan.asMargin = best.worst;
plan.comp = comp;  plan.v = v;  plan.force = pp.k*comp;
plan.Espring = 0.5*pp.k*comp^2;  plan.Ecost = plan.Espring/pp.etaWinch;
plan.vx = v*cos(th);  plan.vz = v*sin(th);
plan.vApp = min(pp.vJump, plan.vx);
plan.fl = fl;  plan.lc = landingCheck(pp, fl.vxTD, fl.vzTD, fl.wTD, fl.phiTD);
plan.mc = minClearance(pp, ob, xl + fl.x, fl.z, fl.phi, zg);
w0  = pp.m*v*sin(th)*pp.thrustOffset/pp.I;
flU = simulateFlight(pp, zg + pp.restZ + comp, v*cos(th), v*sin(th), w0, false, zg, 0);
plan.phiC = wrapPi(fl.phiTD)*180/pi;  plan.phiU = wrapPi(flU.phiTD)*180/pi;
plan.arcX = xl + fl.x;  plan.arcZ = fl.z;
end

%% =====================================================================
%%  EXTENSION 7 : MULTI-ROBOT COOPERATIVE INSPECTION
%% =====================================================================
function M = runTeam(p, env, net, S0)
nR = min(max(env.nRobots, 2), 3);
lanes = [0 -0.40 0.40];  stag = 4.0;
M.nR = nR;  M.lane = lanes(1:nR);  M.t0 = (0:nR-1)*stag;  M.commRange = 3.0;
M.S = cell(1, nR);  M.S{1} = S0;
fprintf(' Multi-robot team: simulating %d spring jumpers (shared map + shared moving-object timeline) ...\n', nR);
for k = 2:nR
    rb = struct('id', k, 'y0', lanes(k), 'yGoal', lanes(k), 'nScan', p.nScan*k, 'quiet', true, ...
                'info', S0.info, 'tOff', M.t0(k), 'dynT0', [S0.dyn.t0]);
    M.S{k} = runSimulation(p, env, net, rb);
    fprintf('   robot R%d: %s, %.1f s, %.1f J\n', k, M.S{k}.status, M.S{k}.t, M.S{k}.E);
end
laneGoal(0);
nO = size(env.obs, 1);  load = zeros(1, nR);  M.owner = ones(nO, 1);
for i = 1:nO
    bid = Inf(1, nR);
    for k = 1:nR
        Fk = M.S{k}.F;  q = find(Fk(:,2) >= env.obs(i,1), 1);
        if ~isempty(q), bid(k) = load(k) + 1.5*abs(Fk(q,17)); end
    end
    [~, w] = min(bid);  M.owner(i) = w;  load(w) = load(w) + 1;
end
tEnd = max(cellfun(@(s) s.F(end,1), M.S) + M.t0);
M.tg = 0:0.05:tEnd;  nT = numel(M.tg);
pos = zeros(nR, nT, 2);
for k = 1:nR
    Fk = M.S{k}.F;  [tu, iu] = unique(Fk(:,1));
    tq = min(max(M.tg - M.t0(k), tu(1)), tu(end));
    pos(k,:,1) = interp1(tu, Fk(iu,2),  tq);
    pos(k,:,2) = interp1(tu, Fk(iu,17), tq);
end
M.pair = nchoosek(1:nR, 2);
M.sep  = zeros(size(M.pair, 1), nT);
for q = 1:size(M.pair, 1)
    a = M.pair(q,1);  b = M.pair(q,2);
    M.sep(q,:) = hypot(pos(a,:,1) - pos(b,:,1), pos(a,:,2) - pos(b,:,2));
end
M.minSep = min(M.sep, [], 2);
end

function Fo = buildFollowers(ax, p, env, S)
Fo = [];
if ~isfield(S, 'M') || isempty(S.M), return; end
M = S.M;  nR = M.nR;
cols = [0.12 0.38 0.80; 0.95 0.45 0.15; 0.25 0.75 0.35];
Fo.n = nR;  Fo.col = cols;
for k = 1:nR
    Fo.lab(k) = text(ax, 0, 0, 0, sprintf('R%d', k), 'Color', cols(k,:)*0.5 + 0.5, 'FontSize', 10, ...
                     'FontWeight', 'bold', 'HorizontalAlignment', 'center');
end
for j = 1:nR-1
    k = j + 1;
    Mb = meshNew();
    Mb = meshBox(Mb, [0 0 0], [p.L p.W p.H], cols(k,:));
    Mb = meshBox(Mb, [0 0 0.012], [p.L - 0.02, p.W + 0.003, 0.010], [0.95 0.97 1.0]);
    Mb = meshBox(Mb, [p.L/2 + 0.02, 0, 0.005], [0.05 0.10 0.05], [0.15 0.15 0.18]);
    Mb = meshBox(Mb, [-0.13 0 p.H/2 + 0.035], [0.02 0.02 0.07], [0.85 0.85 0.88]);
    for hx = [-p.wb/2 p.wb/2]
        for hy = [-1 1]
            Mb = meshBox(Mb, [hx, hy*(p.W/2 + (p.wy - p.W/2)/2), -p.H/2 + 0.012], [0.04, p.wy - p.W/2 + 0.03, 0.014], [0.20 0.21 0.25]);
            Mb = meshCyl(Mb, [hx, hy*p.wy, -p.H/2 - 0.002], 0.024, 0.05, [0.18 0.19 0.22], 12);
        end
    end
    Fo.bodyV{j} = Mb.V;
    Fo.bodyH(j) = meshPatch(ax, Mb, 'FaceLighting', 'flat', 'AmbientStrength', 0.55, 'DiffuseStrength', 0.85);
    for q = 1:4
        Fo.lgU(j,q) = plot3(ax, NaN, NaN, NaN, '-', 'Color', [0.22 0.24 0.29], 'LineWidth', 6);
        Fo.lgL(j,q) = plot3(ax, NaN, NaN, NaN, '-', 'Color', [0.88 0.90 0.94], 'LineWidth', 2.5);
        Fo.lgC(j,q) = plot3(ax, NaN, NaN, NaN, '-', 'Color', cols(k,:)*0.5 + [0.5 0.28 0.05], 'LineWidth', 1.5);
        Fo.kn(j,q)  = plot3(ax, NaN, NaN, NaN, 'o', 'MarkerSize', 4, 'MarkerFaceColor', [0.1 0.9 1], 'MarkerEdgeColor', 'k');
        Fo.pd(j,q)  = plot3(ax, NaN, NaN, NaN, 'o', 'MarkerSize', 7, 'MarkerFaceColor', [0.12 0.12 0.14], 'MarkerEdgeColor', [0.9 0.9 0.9]);
    end
    Fo.trail(j) = plot3(ax, NaN, NaN, NaN, '-', 'Color', cols(k,:), 'LineWidth', 2);
    Fo.link(j)  = plot3(ax, NaN, NaN, NaN, '--', 'Color', [0.4 1 1], 'LineWidth', 1.4);
end
for i = 1:size(env.obs, 1)
    xg = env.obs(i,1) + env.obs(i,2)/2;  zt = terrainZ(env, env.obs(i,1)) + env.obs(i,3) + 0.45;
    text(ax, xg, 0, zt, sprintf('inspector: R%d', M.owner(i)), 'Color', cols(M.owner(i),:)*0.5 + 0.5, ...
         'FontSize', 8, 'FontWeight', 'bold', 'HorizontalAlignment', 'center');
end
end

function drawFollowers(H, p, env, S, t)
Fo = H.fol;
if isempty(Fo), return; end
M = S.M;  P0 = curPose();
P = zeros(Fo.n, 3);
for k = 1:Fo.n
    Fk = M.S{k}.F;
    q = find(Fk(:,1) <= max(t - M.t0(k), 0), 1, 'last');
    if isempty(q), q = 1; end
    r = Fk(q,:);
    P(k,:) = [r(2) r(17) r(3)];
    if k >= 2
        j = k - 1;
        curPose([r(17) r(18)]);
        zbF = r(3) + bobZ(p, r);
        if r(8) == 3 || r(8) == 4, zbF = r(3) - r(6); end
        setV(Fo.bodyH(j), Fo.bodyV{j}, r(2), zbF, r(4));
        [Hw, Kw, Fw, Cw] = legGeometry(p, env, r);
        setLegs(Fo.lgU(j,:), Fo.lgL(j,:), Fo.lgC(j,:), Fo.kn(j,:), Fo.pd(j,:), Hw, Kw, Fw, Cw);
        set(Fo.trail(j), 'XData', Fk(1:q,2), 'YData', Fk(1:q,17), 'ZData', Fk(1:q,3));
    end
    set(Fo.lab(k), 'Position', [r(2) r(17) r(3) + 0.28]);
end
for j = 1:Fo.n - 1
    a = P(j,:);  b = P(j+1,:);
    if hypot(a(1) - b(1), a(2) - b(2)) < M.commRange
        set(Fo.link(j), 'XData', [a(1) b(1)], 'YData', [a(2) b(2)], 'ZData', [a(3) b(3)] + 0.12);
    else
        set(Fo.link(j), 'XData', NaN, 'YData', NaN, 'ZData', NaN);
    end
end
gap = Inf;
for q = 1:size(M.pair, 1)
    gap = min(gap, hypot(P(M.pair(q,1),1) - P(M.pair(q,2),1), P(M.pair(q,1),2) - P(M.pair(q,2),2)));
end
set(H.team, 'String', sprintf('TEAM  R1 x%.1f | R2 x%.1f | R3 x%.1f | min gap %.2f m', ...
    P(1,1), P(min(2,Fo.n),1), P(Fo.n,1), gap));
curPose(P0);
end

%% =====================================================================
%%  SPRING-LEG MODEL : hop-walk gait + telescopic leg with visible coil spring
%% =====================================================================
function W = bwT(L, x, y0, z, phi, yaw)
% body-frame points (n x 3) -> world
c = cos(phi);  s = sin(phi);
bx = L(:,1)*c - L(:,3)*s;
bz = L(:,1)*s + L(:,3)*c;
by = L(:,2);
W = [x + bx*cos(yaw) - by*sin(yaw), y0 + bx*sin(yaw) + by*cos(yaw), z + bz];
end

function B = wbT(W, x, y0, z, phi, yaw)
% world points (n x 3) -> body frame
d1 = W(:,1) - x;  d2 = W(:,2) - y0;  d3 = W(:,3) - z;
bx =  d1*cos(yaw) + d2*sin(yaw);
by = -d1*sin(yaw) + d2*cos(yaw);
c = cos(phi);  s = sin(phi);
B = [bx*c + d3*s, by, -bx*s + d3*c];
end

function dz = bobZ(p, row)
% vertical bounce of the body on its leg springs while walking (drawing only)
dz = 0;
mode = row(8);
if mode == 1 || mode == 10 || mode == 11
    amp = min(abs(row(13))/0.3, 1);
    dg  = row(5)*p.r;
    dz  = -p.bobAmp*amp*(1 - cos(4*pi*dg/p.stride))/2;
end
end

function [Hw, Kw, Fw, Cw] = legGeometry(p, env, row)
% Hip, sleeve-end, foot positions (world, 4 x 3) and coil-spring polylines (cell of n x 3) for one telemetry row.
% leg order: 1 front +y, 2 front -y, 3 hind +y, 4 hind -y.  Diagonal pairs 1&4 and 2&3 step together.
% Each leg is a straight telescopic spring leg: its length is the hip-to-foot distance, so the coil
% visibly compresses in the crouch and stretches at launch.
x = row(2);  z = row(3);  phi = row(4);  wA = row(5);  comp = row(6);  leg = row(7);
mode = row(8);  vx = row(13);  y = row(17);  yaw = row(18);
hz  = -p.H/2 + 0.01;
hipB = [ p.wb/2  p.wy hz;  p.wb/2 -p.wy hz; -p.wb/2  p.wy hz; -p.wb/2 -p.wy hz];
off = [0 0.5 0.5 0];
zc = z + bobZ(p, row);
if mode == 3 || mode == 4, zc = z - comp; end        % body drops while the leg springs compress for the jump
Hw = bwT(hipB, x, y, zc, phi, yaw);
amp = 0;
if mode == 1 || mode == 10 || mode == 11, amp = min(abs(vx)/0.3, 1); end
dg = wA*p.r;                                         % distance walked -> gait phase
a  = p.gaitDuty*p.stride/2;
Kw = zeros(4,3);  Fw = zeros(4,3);  Cw = cell(1,4);
nP = 14*p.nCoil;
th = linspace(0, 2*pi*p.nCoil, nP)';
sn = linspace(0, 1, nP)';
for q = 1:4
    Hb = hipB(q,:);
    if mode == 5                                     % flight: legs retracted, extension decays
        Fb = [Hb(1)*0.75, Hb(2), -0.06 - 0.9*leg];
    elseif mode == 7 || mode == 8                    % tumble / self-right: legs splayed
        Fb = [Hb(1)*1.25, Hb(2), -0.085];
    else
        ph = mod(dg/p.stride + off(q), 1);
        if ph < p.gaitDuty
            sg = ph/p.gaitDuty;
            rx = a*(1 - 2*sg);  lift = 0;
        else
            u  = (ph - p.gaitDuty)/(1 - p.gaitDuty);
            rx = -a*cos(pi*u);  lift = p.stepH*sin(pi*u);
        end
        rx = amp*rx;  lift = amp*lift;
        xf = Hw(q,1) + rx;  yf = Hw(q,2);
        zf = groundHeightXY(env, xf, yf) + p.rf + lift;
        Fb = wbT([xf yf zf], x, y, zc, phi, yaw);
    end
    d  = [Fb(1) - Hb(1), Fb(3) - Hb(3)];
    D  = norm(d);
    if D < 1e-9, d = [0 -1]; D = 1; end
    uu  = d/D;
    Dc  = min(max(D, 0.03), 0.22);                   % telescoping leg length
    nrm = [-uu(2) uu(1)];
    ks  = min(p.legL, 0.6*Dc);                       % guide-sleeve end (collar)
    kn  = [Hb(1) Hb(3)] + uu*ks;
    ft  = [Hb(1) Hb(3)] + uu*Dc;
    sc  = 0.004 + sn*max(Dc - 0.016, 0.004);         % coil runs from the hip to just above the foot pad
    cb  = [Hb(1) + uu(1)*sc + p.coilR*cos(th)*nrm(1), ...
           Hb(2) + p.coilR*sin(th), ...
           Hb(3) + uu(2)*sc + p.coilR*cos(th)*nrm(2)];
    Cw{q}   = bwT(cb, x, y, zc, phi, yaw);
    Kw(q,:) = bwT([kn(1) Hb(2) kn(2)], x, y, zc, phi, yaw);
    Fw(q,:) = bwT([ft(1) Hb(2) ft(2)], x, y, zc, phi, yaw);
end
end

function setLegs(lgU, lgL, lgC, kn, pd, Hw, Kw, Fw, Cw)
for q = 1:4
    set(lgU(q), 'XData', [Hw(q,1) Kw(q,1)], 'YData', [Hw(q,2) Kw(q,2)], 'ZData', [Hw(q,3) Kw(q,3)]);
    set(lgL(q), 'XData', [Kw(q,1) Fw(q,1)], 'YData', [Kw(q,2) Fw(q,2)], 'ZData', [Kw(q,3) Fw(q,3)]);
    C = Cw{q};
    set(lgC(q), 'XData', C(:,1), 'YData', C(:,2), 'ZData', C(:,3));
    set(kn(q),  'XData', Kw(q,1), 'YData', Kw(q,2), 'ZData', Kw(q,3));
    set(pd(q),  'XData', Fw(q,1), 'YData', Fw(q,2), 'ZData', Fw(q,3));
end
end

%% =====================================================================
%%  CONSOLE OUTPUT
%% =====================================================================
function printHeader(p, env)
fprintf('\n=====================================================================\n');
fprintf('  ADAPTIVE SPRING-LEGGED JUMPING INSPECTION ROBOT - DIGITAL TWIN V7 (3D)\n');
fprintf('=====================================================================\n');
fprintf('  mass %.2f kg | I = %.4f kg m^2 | k = %.0f N/m | eta = %.2f\n', p.m, p.I, p.k, p.eta);
fprintf('  max leg-spring travel %.0f mm -> max spring-delivered speed %.2f m/s (asphalt, new spring)\n', 1000*p.maxTravel, ...
        sqrt((p.eta*p.k*p.maxTravel^2 - 2*p.m*p.g*p.maxTravel)/p.m));
fprintf('  RUNNING JUMP: robot keeps walking at %.2f m/s (no stop), springs charge on the move and only add the missing velocity\n', p.vJump);
fprintf('  4 telescopic spring legs: sleeve %.0f mm, coil radius %.0f mm x %d turns, stride %.0f mm, swing lift %.0f mm, duty %.2f\n', ...
        1000*p.legL, 1000*p.coilR, p.nCoil, 1000*p.stride, 1000*p.stepH, p.gaitDuty);
fprintf('  step-over limit %.0f mm | sensor range %.2f m | %d obstacles, %.1f m course\n', ...
        1000*p.hRollMax, p.senseRange, size(env.obs,1), env.xEnd);
fprintf('  battery %.0f J + solar %.1f W | spring fatigue %.0f %%/jump | MC trials %d\n', ...
        p.Ebat, p.solarW, 100*p.fatigue, p.mcN);
fprintf('  extensions: AI decision %s | moving obstacles %s (%d, of which %d moving rocks) | fault injection %s (%d) | route mode %s\n', ...
        onoff(env.aiOn), onoff(env.dynOn), numel(env.dyn), sum([env.dyn.rock]), onoff(env.faultsOn), numel(env.faults), upper(env.routeMode));
fprintf('              adaptive spring %s | multi-robot %s (%d robots)\n', onoff(env.asOn), onoff(env.multiOn), env.nRobots);
fprintf('              terrain classification %s | collision prediction %s\n', onoff(env.terrOn), onoff(env.colOn));
fprintf('              hazard ranking ON | energy forecast ON\n');
fprintf('---------------------------------------------------------------------\n');
end

function s = onoff(b)
if b, s = 'ON'; else, s = 'OFF'; end
end

function printSummary(p, env, S, CM)
fprintf('---------------------------------------------------------------------\n');
fprintf(' OBSTACLE SUMMARY  (AI proposal -> verified decision, energies of the three options)\n');
for i = 1:numel(S.ev)
    e = S.ev(i);
    aiS = '-';
    if e.cAI > 0, aiS = sprintf('%s %3.0f%%', aiName(e.cAI), 100*e.pAI); end
    fprintf(' #%d %-14s h%4.0f w%4.0f hw%5.0f mm | AI %-10s -> %-6s | Es %-8s Ej %-8s Ea %-8s', i, env.obsName{i}, ...
            1000*e.hM, 1000*e.wM, 1000*e.hwM, aiS, e.decision, fmtE(e.Eroll), fmtE(e.Ejump), fmtE(e.Eavoid));
    if ~isempty(e.plan)
        pl = e.plan;
        fprintf(' | v %.2f m/s %2.0f deg (run-up %.2f), crouch %4.1f mm %4.0f N, land %5.1f deg', pl.v, pl.theta*180/pi, pl.vApp, ...
                1000*pl.comp, pl.force, e.landPhi);
    end
    fprintf('\n');
end
if S.kick.happened && ~isempty(S.kick.sr)
    fprintf(' Self-righting: hop %.2f m/s, L = %.3f kg m2/s, spring %.1f mm, landing pitch %.1f deg\n', ...
            S.kick.sr.vz, S.kick.sr.L, 1000*S.kick.sr.comp, wrapPi(S.kick.sr.fl.phiTD)*180/pi);
end
dist = max(S.x - 0.35, 0.1);
fprintf(' Total energy used: %.1f J   |   mission time: %.1f s   |   status: %s\n', S.E, S.t, S.status);
fprintf(' Cost of transport: %.2f   |   battery left: %.0f %% (solar harvested %.1f J)\n', ...
        S.E/(p.m*p.g*dist), 100*batterySOC(p, S.t, S.E), p.solarW*S.t);
fprintf(' Spring health: %.1f %% after %d jumps\n', 100*S.health, S.jumps);
for q = 1:numel(S.MC.idx)
    fprintf(' Monte-Carlo landing success, obstacle #%d: %5.1f %%  (%d trials)\n', ...
            S.MC.idx(q), 100*S.MC.prob(q), S.MC.n);
end
if ~isempty(CM)
    fprintf(' Capability map: %d of %d tested obstacle sizes are jumpable (>step-over limit)\n', ...
            sum(CM.mode(:) == 2), sum(CM.mode(:) >= 2));
end
A = S.R0.shortest;  B = S.R0.safest;
fprintf(' Route planner (initial): shortest %.2f m, %.1f J, risk %.1f | safest %.2f m, %.1f J, risk %.1f | followed: %s\n', ...
        A.len, A.E, A.risk, B.len, B.E, B.risk, upper(env.routeMode));
fprintf(' Route re-plans during mission: %d\n', numel(S.routes) - 1);
if env.dynOn
    for i = 1:numel(S.dyn)
        nWait = sum(S.F(:,8) == 11)*0.01;
        fprintf(' Moving %s #%d (%s): triggered at t = %.2f s, final state %d\n', ...
                ternary(env.dyn(i).rock, 'rock', 'object'), i, env.dyn(i).name, S.dyn(i).t0, S.dyn(i).state);
        if i == numel(S.dyn), fprintf(' Total time spent yielding to traffic: %.1f s\n', nWait); end
    end
end
if env.faultsOn
    for i = 1:numel(env.faults)
        f = env.faults(i);
        fprintf(' Fault scheduled: %-6s x = %.1f ... %.1f m\n', f.type, f.x0, f.x1);
    end
end
fprintf('---------------------------------------------------------------------\n');
end

function printExtra(p, env, S) %#ok<INUSL>
fprintf(' EXTENSION 6 - ADAPTIVE SPRING CONTROL: %s\n', onoff(env.asOn));
for i = find([S.ev.executed])
    pl = S.ev(i).plan;
    if isfield(pl, 'asMul')
        fprintf('  #%d %-14s crouch %.1f -> %.1f mm (x%.2f), F = %.0f N, worst-case landing margin %.0f mm\n', ...
                i, env.obsName{i}, 1000*pl.comp0, 1000*pl.comp, pl.asMul, pl.force, 1000*pl.asMargin);
    end
end
if ~isempty(S.M)
    M = S.M;
    fprintf(' EXTENSION 7 - MULTI-ROBOT INSPECTION: %d robots, start stagger %.1f s, comm range %.1f m\n', M.nR, M.t0(2), M.commRange);
    for k = 1:M.nR
        fprintf('  R%d lane %+.2f m: %s, %.1f J, %.1f s, inspects obstacles %s\n', k, M.lane(k), M.S{k}.status, ...
                M.S{k}.E, M.S{k}.t, mat2str(find(M.owner == k)'));
    end
    for q = 1:size(M.pair, 1)
        fprintf('  min separation R%d-R%d: %.2f m %s\n', M.pair(q,1), M.pair(q,2), M.minSep(q), ...
                ternary(M.minSep(q) < 0.35, '(NEAR MISS)', '(safe)'));
    end
end
fprintf('---------------------------------------------------------------------\n');
fprintf(' Starting 3D animation...  keys: SPACE pause | S slow | R replay | C camera | Q quit\n\n');
end

function printTC(p, env, S)
fprintf(' EXTENSION 8 - TERRAIN CLASSIFICATION: %s\n', onoff(env.terrOn));
if ~isempty(S.T)
    fprintf('  online accuracy %.1f %%\n', 100*mean(S.T(:,1) == S.T(:,2)));
end
fprintf(' EXTENSION 9 - COLLISION PREDICTION: %s\n', onoff(env.colOn));
if ~isempty(S.C)
    C = S.C;  F = S.F;  nReal = 0;
    if env.dynOn
        for i = 1:numel(env.dyn)
            d = env.dyn(i);
            yd = arrayfun(@(t) dynY(d, S.dyn(i).t0, t), F(:,1));
            nReal = nReal + sum(abs(F(:,2) - d.x) < (d.len + p.L)/2 & abs(F(:,17) - yd) < d.hw + p.robotHW);
        end
    end
    k1 = find(C(:,2) >= 1, 1);
    if isempty(k1)
        fprintf('  no threat tracked, ');
    else
        fprintf('  first threat at t = %.2f s, ', F(k1,1));
    end
    fprintf('predicted-collision frames %d, actual moving-object collisions %d\n', sum(C(:,2) >= 2), nReal);
end
fprintf('---------------------------------------------------------------------\n');
end

%% =====================================================================
%%  3D ANIMATION LOOP
%% =====================================================================
function playAnimation(p, env, S)
F = S.F;  N = size(F,1);
fig = figure('Name','Spring-Legged Jumping Inspection Robot - Digital Twin V7 (3D)','NumberTitle','off', ...
             'Position',[20 20 1440 860],'Color',[0.04 0.05 0.08],'KeyPressFcn',@onKey);
setappdata(fig, 'ctl', struct('paused',false,'slow',false,'restart',false,'quit',false,'cam',1));
H = buildScene3D(p, env, S, fig);

tw = tic;  simClock = 0;  lastW = 0;  finished = false;
try
    while ishandle(fig)
        ctl = getappdata(fig, 'ctl');
        if ctl.quit, break; end
        if ctl.restart
            ctl.restart = false;  setappdata(fig, 'ctl', ctl);
            simClock = 0;  finished = false;
        end
        wNow = toc(tw);  dW = wNow - lastW;  lastW = wNow;
        if ~ctl.paused && ~finished
            sp = 1;
            if ctl.slow, sp = 0.25; end
            simClock = simClock + dW*sp;
        end
        k = find(F(:,1) <= simClock, 1, 'last');
        if isempty(k), k = 1; end
        if k >= N, k = N; finished = true; end
        drawFrame3D(H, p, env, S, k, ctl, finished, wNow);
        drawnow limitrate;
        pause(0.005);
    end
catch ME
    if ishandle(fig), rethrow(ME); end
end
if ishandle(fig), close(fig); end
end

function onKey(src, evt)
ctl = getappdata(src, 'ctl');
switch lower(evt.Key)
    case 'space', ctl.paused = ~ctl.paused;
    case 's',     ctl.slow = ~ctl.slow;
    case 'r',     ctl.restart = true;  ctl.paused = false;
    case 'c',     ctl.cam = mod(ctl.cam, 5) + 1;
    case 'q',     ctl.quit = true;
end
setappdata(src, 'ctl', ctl);
end

%% =====================================================================
%%  3D MESH HELPERS
%% =====================================================================
function M = meshNew()
M.V = zeros(0,3);  M.F = zeros(0,4);  M.C = zeros(0,3);
end

function M = meshBox(M, c, s, col)
h = s/2;
V = [c(1) + h(1)*[-1 1 1 -1 -1 1 1 -1]', c(2) + h(2)*[-1 -1 1 1 -1 -1 1 1]', ...
     c(3) + h(3)*[-1 -1 -1 -1 1 1 1 1]'];
Fc = [1 2 3 4; 5 8 7 6; 1 5 6 2; 2 6 7 3; 3 7 8 4; 4 8 5 1];
n0 = size(M.V,1);
M.V = [M.V; V];  M.F = [M.F; Fc + n0];  M.C = [M.C; repmat(col, 6, 1)];
end

function M = meshQuad(M, P, col)
n0 = size(M.V,1);
M.V = [M.V; P];  M.F = [M.F; n0 + [1 2 3 4]];  M.C = [M.C; col];
end

function M = meshSurf(M, X, Y, Z, col)
[Fc, V] = surf2patch(X, Y, Z);
n0 = size(M.V,1);
M.V = [M.V; V];  M.F = [M.F; Fc + n0];  M.C = [M.C; repmat(col, size(Fc,1), 1)];
end

function M = meshSphere(M, c, r, col, sc, n)
[X, Y, Z] = sphere(n);
M = meshSurf(M, c(1) + r*sc(1)*X, c(2) + r*sc(2)*Y, c(3) + r*sc(3)*Z, col);
end

function y = sgnPow(a, e)
y = sign(a).*abs(a).^e;
end

function M = meshRock(M, c, s, col, seed, n)
rsr = RandStream('mt19937ar', 'Seed', seed);
ph  = 2*pi*rand(rsr, 1, 4);
[U, V] = meshgrid(linspace(-pi, pi, n + 1), linspace(-pi/2, pi/2, floor(n/2) + 1));
e  = 0.55;
cx = sgnPow(cos(V), e).*sgnPow(cos(U), e);
cy = sgnPow(cos(V), e).*sgnPow(sin(U), e);
cz = sgnPow(sin(V), e);
nz = 0.10*(sin(3*U + ph(1)).*cos(2*V + ph(2)) + 0.6*sin(5*U + 3*V + ph(3)) + 0.4*sin(7*U - 2*V + ph(4)));
f  = 1 + nz;
fz = f;
top = cz > 0;
fz(top) = 0.99*(1 + 0.10*nz(top));
X = c(1) + 0.5*s(1)*cx.*f;
Y = c(2) + 0.5*s(2)*cy.*f;
Z = c(3) + 0.5*s(3)*cz.*fz;
[Fc, Vv] = surf2patch(X, Y, Z);
zc = (Vv(Fc(:,1),3) + Vv(Fc(:,2),3) + Vv(Fc(:,3),3) + Vv(Fc(:,4),3))/4;
t  = min(max((zc - (c(3) - s(3)/2))/max(s(3), 1e-6), 0), 1);
shade = 0.72 + 0.38*t;
grain = 1 + 0.16*(rand(rsr, size(Fc,1), 1) - 0.5);
Cc = min(max(bsxfun(@times, col, shade.*grain), 0), 1);
n0 = size(M.V, 1);
M.V = [M.V; Vv];  M.F = [M.F; Fc + n0];  M.C = [M.C; Cc];
end

function M = meshCone(M, c, r, h, col, n)
[X, Y, Z] = cylinder([r 0], n);
M = meshSurf(M, c(1) + X, c(2) + Y, c(3) + h*Z, col);
end

function M = meshCyl(M, c, r, h, col, n)
[X, Y, Z] = cylinder(r, n);
M = meshSurf(M, c(1) + X, c(2) + Y, c(3) + h*Z, col);
end

function h = meshPatch(ax, M, varargin)
h = patch('Parent', ax, 'Vertices', M.V, 'Faces', M.F, 'FaceVertexCData', M.C, ...
          'FaceColor', 'flat', 'EdgeColor', 'none', varargin{:});
end

function y = sstep(a, b, x)
t = min(max((x - a)./(b - a), 0), 1);
y = t.*t.*(3 - 2*t);
end

function Hh = hill3D(env, X, Y)
ay  = abs(Y);
e   = sstep(0.9, 2.6, ay);
rip = 0.045*(sin(1.7*X + 0.9*ay) + sin(0.8*X - 1.3*ay + 1));
big = 1.1*sstep(7, 28, ay).*(0.55 + 0.45*sin(0.31*X + 0.17*ay + 2));
mY  = 13*sstep(30, 60, ay).*(0.55 + 0.45*sin(0.13*X + 1.1).*cos(0.23*ay));
dx  = max(max(-X - 10, X - (env.xEnd + 10)), 0);
mX  = 11*sstep(22, 52, dx).*(0.6 + 0.4*sin(0.3*Y + 0.8));
mtn = max(mY, mX).*(0.85 + 0.3*abs(sin(0.45*X + 0.33*ay)));
Hh  = e.*(rip + big) + mtn;
end

function [X, Y, Z] = b2w(Lc, x, z, phi)
P = curPose();  y0 = P(1);  yaw = P(2);
c = cos(phi);  s = sin(phi);
bx = Lc(:,1)*c - Lc(:,3)*s;
bz = Lc(:,1)*s + Lc(:,3)*c;
by = Lc(:,2);
cyw = cos(yaw);  syw = sin(yaw);
X = x + bx*cyw - by*syw;
Y = y0 + bx*syw + by*cyw;
Z = z + bz;
end

function V = rotY(V, a)
c = cos(a);  s = sin(a);
V = [V(:,1)*c + V(:,3)*s, V(:,2), -V(:,1)*s + V(:,3)*c];
end

function V = rotX(V, a)
c = cos(a);  s = sin(a);
V = [V(:,1), V(:,2)*c - V(:,3)*s, V(:,2)*s + V(:,3)*c];
end

function setV(h, Vl, x, z, phi)
[X, Y, Z] = b2w(Vl, x, z, phi);
set(h, 'Vertices', [X Y Z]);
end

function [V, Fc, Cc] = wheelMesh(r, w, n, tireA, tireB, rimA, rimB)
% disc mesh (used here for the reaction wheel)
t = (0:n-1)'*2*pi/n;
A = [r*cos(t),  w/2*ones(n,1), r*sin(t)];
B = [r*cos(t), -w/2*ones(n,1), r*sin(t)];
V = [A; B; 0 w/2 0; 0 -w/2 0];
Fc = zeros(3*n, 4);  Cc = zeros(3*n, 3);
for i = 1:n
    j = mod(i, n) + 1;
    Fc(i,:)     = [i, j, n + j, n + i];
    Fc(n+i,:)   = [2*n + 1, i, j, 2*n + 1];
    Fc(2*n+i,:) = [2*n + 2, n + j, n + i, 2*n + 2];
    if mod(i,2) == 0, Cc(i,:) = tireA; else, Cc(i,:) = tireB; end
    if mod(floor((i-1)/(n/6)), 2) == 0, rc = rimA; else, rc = rimB; end
    Cc(n+i,:) = rc;  Cc(2*n+i,:) = rc;
end
end

function V = boxVerts(c, s)
h = s/2;
V = [c(1) + h(1)*[-1 1 1 -1 -1 1 1 -1]', c(2) + h(2)*[-1 -1 1 1 -1 -1 1 1]', ...
     c(3) + h(3)*[-1 -1 -1 -1 1 1 1 1]'];
end

function Fc = boxFaces()
Fc = [1 2 3 4; 5 8 7 6; 1 5 6 2; 2 6 7 3; 3 7 8 4; 4 8 5 1];
end

%% =====================================================================
%%  3D SCENE
%% =====================================================================
function H = buildScene3D(p, env, S, fig)
H.fig = fig;
fixedFont = get(0, 'FixedWidthFontName');
cy = [0.25 0.90 1.00];

ax = axes('Parent', fig, 'Position', [0.005 0.24 0.685 0.755], 'Color', [0.55 0.7 0.9]);
hold(ax, 'on');  H.ax = ax;
set(ax, 'XLim', [-125 135], 'YLim', [-125 125], 'ZLim', [-10 125], 'DataAspectRatio', [1 1 1], ...
        'Projection', 'perspective', 'CameraViewAngleMode', 'manual', 'AmbientLightColor', [0.42 0.44 0.50]);
sunDir = [0.86 -0.30 0.33];  sunDir = sunDir/norm(sunDir);
light('Parent', ax, 'Style', 'infinite', 'Position', sunDir, 'Color', [1.00 0.92 0.80]);
light('Parent', ax, 'Style', 'infinite', 'Position', [-0.5 -0.8 0.35], 'Color', [0.42 0.48 0.62]);
light('Parent', ax, 'Style', 'infinite', 'Position', [-1.0 0.2 0.5], 'Color', [0.28 0.28 0.32]);

addSky(ax, sunDir);
sceneH = @(x, y) terrainZ(env, x) + hill3D(env, x, y);
addTerrain(ax, env);

rs = RandStream('mt19937ar', 'Seed', 5);
Ms = meshNew();  Mg = meshNew();  Mh = meshNew();

rockPal = [0.50 0.48 0.46; 0.45 0.42 0.40; 0.58 0.50 0.40; 0.52 0.52 0.54; ...
           0.40 0.36 0.32; 0.55 0.52 0.48; 0.47 0.46 0.44; 0.62 0.55 0.42; ...
           0.42 0.37 0.31; 0.36 0.35 0.35; 0.50 0.45 0.40; 0.44 0.43 0.45];
for i = 1:size(env.obs,1)
    xo = env.obs(i,1);  wo = env.obs(i,2);  ho = env.obs(i,3);  zg = terrainZ(env, xo);
    yw = env.obsHalfW(i);
    rr = RandStream('mt19937ar', 'Seed', 100 + i);
    nSeg = max(1, ceil(2*yw/0.45));  segW = 2*yw/nSeg;
    for sg = 1:nSeg
        yc = -yw + (sg - 0.5)*segW;
        sz = [wo*(1.00 + 0.06*rand(rr)), segW*1.30, ho];
        colR = min(rockPal(i,:)*(0.93 + 0.14*rand(rr)), 1);
        Ms = meshRock(Ms, [xo + wo/2, yc, zg + ho/2], sz, colR, 100*i + sg, 16);
    end
    for q = 1:3
        sgn = [-1 1 1];
        yq = sgn(q)*(yw + 0.06 + 0.08*rand(rr));  xq = xo + wo*rand(rr);
        rpb = 0.015 + 0.02*rand(rr);
        Ms = meshRock(Ms, [xq, yq, zg + 0.5*rpb], [2.4*rpb, 2*rpb, 1.2*rpb], ...
                      min(rockPal(i,:)*1.1, 1), 700 + 3*i + q, 8);
    end
    for sgn = [-1 1]
        cB = [xo + wo/2, sgn*yw, zg + ho + 0.02];
        Mg = meshSphere(Mg, cB, 0.013, [1 0.15 0.1], [1 1 1], 6);
        Mh = meshSphere(Mh, cB, 0.055, [1 0.25 0.15], [1 1 1], 6);
    end
    xg = xo + wo/2;  zt = zg + ho + 0.32;  yg = yw + 0.12;
    plot3(ax, xg*ones(1,4), [-yg -yg yg yg], [zg zt zt zg], '-', 'Color', cy, 'LineWidth', 1.6);
    patch('Parent', ax, 'XData', xg*ones(4,1), 'YData', [-yg; -yg; yg; yg], 'ZData', [zg; zt; zt; zg], ...
          'FaceColor', cy, 'FaceAlpha', 0.06, 'EdgeColor', 'none', 'FaceLighting', 'none');
    text(ax, xg, 0, zt + 0.05, sprintf('#%d %s   h = %.0f mm', i, env.obsName{i}, 1000*ho), ...
         'Color', [0.7 1 1], 'FontSize', 8, 'FontWeight', 'bold', 'HorizontalAlignment', 'center', ...
         'Interpreter', 'none');
end

cp = [0.35 0 0.003];
Mg = meshBox(Mg, cp + [0 0.22 0], [0.5 0.012 0.003], cy);
Mg = meshBox(Mg, cp - [0 0.22 0], [0.5 0.012 0.003], cy);
Mg = meshBox(Mg, cp + [0.25 0 0], [0.012 0.45 0.003], cy);
Mg = meshBox(Mg, cp - [0.25 0 0], [0.012 0.45 0.003], cy);
[Ms, Mg] = meshArch(Ms, Mg, -0.55, 0, 1.15, 0.75);
[Ms, Mg] = meshArch(Ms, Mg, env.xEnd + 1.0, 0, 1.15, 0.75);
text(ax, -0.55, 0, 0.88, 'START', 'Color', [0.6 1 0.7], 'FontSize', 13, 'FontWeight', 'bold', ...
     'HorizontalAlignment', 'center');
text(ax, env.xEnd + 1.0, 0, 0.88, 'FINISH', 'Color', [1 0.85 0.4], 'FontSize', 13, 'FontWeight', 'bold', ...
     'HorizontalAlignment', 'center');

for xl = 0:2.5:env.xEnd
    side = 1 - 2*mod(round(xl/2.5), 2);
    yl = side*1.15;  zb = sceneH(xl, yl);
    Ms = meshBox(Ms, [xl, yl, zb + 0.28], [0.018 0.018 0.56], [0.18 0.19 0.22]);
    Ms = meshBox(Ms, [xl, yl - side*0.055, zb + 0.565], [0.016 0.11 0.014], [0.18 0.19 0.22]);
    hp = [xl, yl - side*0.11, zb + 0.55];
    Mg = meshSphere(Mg, hp, 0.022, [1.0 0.93 0.72], [1 1 1], 6);
    Mh = meshSphere(Mh, hp, 0.12, [1.0 0.82 0.45], [1 1 1], 6);
end

bx0 = -3.5;  by0 = 3.3;  bz0 = sceneH(bx0, by0);
Ms = meshBox(Ms, [bx0, by0, bz0 + 0.4], [1.7 1.1 0.8], [0.88 0.90 0.93]);
Ms = meshBox(Ms, [bx0, by0, bz0 + 0.83], [1.9 1.3 0.06], [0.17 0.19 0.23]);
Ms = meshBox(Ms, [bx0 + 0.55, by0, bz0 + 1.12], [0.04 0.04 0.5], [0.8 0.8 0.84]);
Mg = meshBox(Mg, [bx0 - 0.1, by0 - 0.555, bz0 + 0.45], [1.3 0.01 0.32], [0.35 0.80 1.0]);
Mg = meshBox(Mg, [bx0 + 0.55, by0 - 0.555, bz0 + 0.25], [0.28 0.01 0.5], cy);
Mg = meshSphere(Mg, [bx0 + 0.55, by0, bz0 + 1.4], 0.03, [1 0.2 0.15], [1 1 1], 6);
Mh = meshSphere(Mh, [bx0 + 0.55, by0, bz0 + 1.4], 0.14, [1 0.25 0.15], [1 1 1], 6);
text(ax, bx0, by0 - 0.58, bz0 + 0.68, 'BASE STATION', 'Color', [0.1 0.12 0.2], 'FontSize', 8, ...
     'FontWeight', 'bold', 'HorizontalAlignment', 'center');

for r = 0:1
    for c = 0:20
        x0 = 2.0 + c*1.25;  y0 = 3.7 + r*1.35;  zb = sceneH(x0, y0) + 0.12;
        for q = 1:6
            xa = x0 + (q-1)*0.18;  xb = xa + 0.18;
            if mod(q,2) == 0, col = [0.05 0.12 0.36]; else, col = [0.08 0.17 0.44]; end
            P = [xa y0 zb; xb y0 zb; xb y0 + 0.55 zb + 0.28; xa y0 + 0.55 zb + 0.28];
            Ms = meshQuad(Ms, P, col);
        end
        Ms = meshBox(Ms, [x0 + 0.54, y0 + 0.3, zb*0.5], [0.025 0.025 zb], [0.5 0.5 0.55]);
    end
end

turb = [1.5 18; 9.0 24; 15.5 14.5; -6 21; 22 17; 29 22];
H.hub = zeros(size(turb,1), 3);
for j = 1:size(turb,1)
    zT = sceneH(turb(j,1), turb(j,2));  ht = 6;
    [X, Y, Z] = cylinder([0.11 0.05], 10);
    Ms = meshSurf(Ms, turb(j,1) + X, turb(j,2) + Y, zT + ht*Z, [0.93 0.94 0.96]);
    Ms = meshBox(Ms, [turb(j,1), turb(j,2) + 0.05, zT + ht + 0.05], [0.24 0.55 0.22], [0.88 0.89 0.92]);
    H.hub(j,:) = [turb(j,1), turb(j,2) - 0.32, zT + ht + 0.05];
    Ms = meshSphere(Ms, H.hub(j,:), 0.1, [0.9 0.9 0.93], [1 1 1], 6);
end

for i = 1:180
    x = -12 + (env.xEnd + 26)*rand(rs);
    y = (2*(rand(rs) > 0.5) - 1)*(2.2 + 16*rand(rs)^1.6);
    zb = sceneH(x, y);  h = 0.9 + 1.4*rand(rs);
    if rand(rs) < 0.82
        gcol = [0.04 0.26 0.11] + 0.08*rand(rs, 1, 3).*[0.4 1 0.4];
        Ms = meshCyl(Ms, [x y zb], 0.03*h, 0.25*h, [0.30 0.20 0.12], 6);
        Ms = meshCone(Ms, [x y zb + 0.18*h], 0.34*h, 0.55*h, gcol, 8);
        Ms = meshCone(Ms, [x y zb + 0.50*h], 0.26*h, 0.50*h, gcol*1.10, 8);
        Ms = meshCone(Ms, [x y zb + 0.80*h], 0.17*h, 0.45*h, gcol*1.2, 8);
    else
        cols = [0.95 0.55 0.15; 0.85 0.75 0.20; 0.25 0.60 0.40];
        Ms = meshCyl(Ms, [x y zb], 0.03*h, 0.45*h, [0.35 0.24 0.15], 6);
        Ms = meshSphere(Ms, [x y zb + 0.62*h], 0.32*h, cols(randi(rs, 3), :), [1 1 0.85], 7);
    end
end
for i = 1:70
    if i <= 45, x = 9 + 4.5*rand(rs); else, x = 24 + 3*rand(rs); end
    y = (2*(rand(rs) > 0.5) - 1)*(1.15 + 2.2*rand(rs));
    Ms = meshSphere(Ms, [x y sceneH(x, y) + 0.04], 0.07 + 0.09*rand(rs), [0.12 0.36 + 0.12*rand(rs) 0.12], [1 1 0.7], 6);
end
for i = 1:85
    x = -3 + (env.xEnd + 6)*rand(rs);  y = (2*(rand(rs) > 0.5) - 1)*(1.2 + 7*rand(rs));
    sz = 0.05 + 0.30*rand(rs)^2;  g = 0.38 + 0.25*rand(rs);
    Ms = meshSphere(Ms, [x y sceneH(x, y) + 0.2*sz], sz, [g g*0.96 g*0.9], [1 0.8 0.55], 6);
end

% ---- MORE ROCKS: large boulders and rock clusters beside the lane ----
% separate random stream so the rest of the scene is unchanged; all kept outside the lane (|y| > 1.4)
rb2 = RandStream('mt19937ar', 'Seed', 77);
for i = 1:60                                                  % scattered boulders
    x  = -4 + (env.xEnd + 8)*rand(rb2);
    y  = (2*(rand(rb2) > 0.5) - 1)*(1.4 + 6.0*rand(rb2)^1.5);
    sz = 0.18 + 0.55*rand(rb2)^1.5;
    g  = 0.34 + 0.26*rand(rb2);
    colB = [g, g*0.96, g*0.90];
    if rand(rb2) < 0.25, colB = [0.55 0.45 0.34]*(0.85 + 0.3*rand(rb2)); end   % some sandstone-coloured rocks
    dims = [sz*(1.0 + 0.6*rand(rb2)), sz*(0.8 + 0.5*rand(rb2)), 0.6*sz];
    Ms = meshRock(Ms, [x, y, sceneH(x, y) + 0.25*sz], dims, min(colB, 1), 900 + i, 14);
end
for i = 1:22                                                  % rock clusters (one big rock + small ones)
    x0 = -2 + (env.xEnd + 4)*rand(rb2);
    y0 = (2*(rand(rb2) > 0.5) - 1)*(1.5 + 3.5*rand(rb2));
    g  = 0.36 + 0.24*rand(rb2);
    for q = 1:4
        x  = x0 + 0.35*randn(rb2);
        y  = y0 + 0.35*randn(rb2);
        if abs(y) < 1.3, y = sign(y0)*1.3; end                % keep the lane clear
        sz = (0.30 - 0.05*q)*(0.7 + 0.6*rand(rb2)) + 0.06;
        Ms = meshRock(Ms, [x, y, sceneH(x, y) + 0.22*sz], [1.3*sz, sz, 0.65*sz], ...
                      min([g g*0.95 g*0.88]*(0.9 + 0.2*rand(rb2)), 1), 1200 + 4*i + q, 12);
    end
end

hS = meshPatch(ax, Ms, 'FaceLighting', 'flat', 'AmbientStrength', 0.55, 'DiffuseStrength', 0.85, ...
               'SpecularStrength', 0.12);  %#ok<NASGU>
meshPatch(ax, Mg, 'FaceLighting', 'none');
meshPatch(ax, Mh, 'FaceLighting', 'none', 'FaceAlpha', 0.14);

nf = 160;
xf = 9.4 + 3.3*rand(rs, nf, 1);  yf = (2*(rand(rs, nf, 1) > 0.5) - 1).*(1.0 + 3*rand(rs, nf, 1));
plot3(ax, xf(1:80), yf(1:80), sceneH(xf(1:80), yf(1:80)) + 0.03, '.', 'Color', [1 0.85 0.2], 'MarkerSize', 9);
plot3(ax, xf(81:end), yf(81:end), sceneH(xf(81:end), yf(81:end)) + 0.03, '.', 'Color', [1 0.45 0.65], 'MarkerSize', 9);
nt = 220;
xt = [9.4 + 3.4*rand(rs, nt, 1); 24 + 3*rand(rs, 120, 1)];  nt = numel(xt);
yt = (2*(rand(rs, nt, 1) > 0.5) - 1).*(0.6 + 3.5*rand(rs, nt, 1));
zt0 = sceneH(xt, yt);
gx = [xt, xt + 0.012, NaN(nt,1)]';  gy = [yt, yt, NaN(nt,1)]';  gz = [zt0, zt0 + 0.045, NaN(nt,1)]';
plot3(ax, gx(:), gy(:), gz(:), '-', 'Color', [0.12 0.50 0.12], 'LineWidth', 1);
xpb = [5 + 3*rand(rs, 160, 1); 20.5 + 2*rand(rs, 120, 1)];  ypb = -0.9 + 1.8*rand(rs, 280, 1);
plot3(ax, xpb, ypb, terrainZ(env, xpb) + 0.003, '.', 'Color', [0.28 0.25 0.23], 'MarkerSize', 6);
xsd = [12.5 + 4.5*rand(rs, 220, 1); 27 + 4.5*rand(rs, 220, 1)];  ysd = -0.9 + 1.8*rand(rs, 440, 1);
plot3(ax, xsd, ysd, terrainZ(env, xsd) + 0.003, '.', 'Color', [0.70 0.58 0.30], 'MarkerSize', 5);

H.routeS = plot3(ax, S.R0.shortest.x, S.R0.shortest.y, terrainZ(env, S.R0.shortest.x) + 0.015, ':', ...
                 'Color', [1 0.6 0.15], 'LineWidth', 2.2);
H.routeA = plot3(ax, S.R0.safest.x, S.R0.safest.y, terrainZ(env, S.R0.safest.x) + 0.018, '-', ...
                 'Color', [0.3 1 0.4], 'LineWidth', 1.6);
H.routeC = plot3(ax, NaN, NaN, NaN, '-', 'Color', [1 1 1], 'LineWidth', 1.2);
H.cpLine = plot3(ax, NaN, NaN, NaN, '-', 'Color', [1 0.2 0.2], 'LineWidth', 3);

H.dynH = [];  H.dynV = {};  H.dynEst = [];
if env.dynOn
    for i = 1:numel(env.dyn)
        d = env.dyn(i);
        if d.rock
            Mr = meshRock(meshNew(), [0 0 0], [d.len 2*d.hw 2*d.hw], [0.52 0.49 0.45], 300 + i, 14);
            Vb = Mr.V;  Fb = Mr.F;  Cb = Mr.C;
        else
            Vb = [boxVerts([0 0 0], [d.len 2*d.hw 0.12]); boxVerts([0 0 0.085], [0.05 0.05 0.05])];
            Fb = [boxFaces(); boxFaces() + 8];
            Cb = [repmat([1.0 0.55 0.10], 6, 1); repmat([1 0.15 0.1], 6, 1)];
        end
        H.dynV{i} = Vb;
        H.dynH(i) = patch('Parent', ax, 'Vertices', Vb, 'Faces', Fb, 'FaceVertexCData', Cb, 'FaceColor', 'flat', ...
                          'EdgeColor', 'none', 'FaceLighting', 'flat', 'AmbientStrength', 0.65);
        zr = terrainZ(env, d.x) + 0.006;
        plot3(ax, d.x*[1 1], [d.y0 -d.y0], zr*[1 1], ':', 'Color', [1 0.7 0.2], 'LineWidth', 1.3);
        text(ax, d.x, d.y0, zr + 0.36, sprintf('MOVING #%d: %s', i, d.name), 'Color', [1 0.85 0.4], ...
             'FontSize', 8, 'FontWeight', 'bold', 'HorizontalAlignment', 'center', 'Interpreter', 'none');
        H.dynEst(i) = plot3(ax, NaN, NaN, NaN, 'o', 'Color', [1 0.3 0.9], 'MarkerSize', 9, 'LineWidth', 1.6);
    end
end

nTb = size(turb,1);
H.bladeF = zeros(3*nTb, 4);
for j = 1:nTb, H.bladeF(3*(j-1) + (1:3), :) = 12*(j-1) + [1 2 3 4; 5 6 7 8; 9 10 11 12]; end
H.blade = patch('Parent', ax, 'Vertices', zeros(12*nTb, 3), 'Faces', H.bladeF, 'FaceColor', [0.95 0.96 0.98], ...
                'EdgeColor', 'none', 'FaceLighting', 'flat', 'AmbientStrength', 0.7);
Mc = meshNew();
for i = 1:10
    c0 = [-60 + 160*rand(rs), -90 + 180*rand(rs), 32 + 22*rand(rs)];
    for j = 1:4
        off = [8*randn(rs), 8*randn(rs), 1.5*randn(rs)];
        Mc = meshSphere(Mc, c0 + off, 5 + 4*rand(rs), [1 0.97 0.95], [1.3 1 0.38], 6);
    end
end
H.cloudV0 = Mc.V;
H.cloud = meshPatch(ax, Mc, 'FaceLighting', 'none', 'FaceAlpha', 0.85);

H.R = buildRobot(ax, p, env);
H.fol = buildFollowers(ax, p, env, S);

H.dustDir = [2*rand(rs, 32, 1) - 1, 2*rand(rs, 32, 1) - 1, rand(rs, 32, 1), rand(rs, 32, 1)];
H.dust = plot3(ax, NaN, NaN, NaN, '.', 'Color', [0.80 0.68 0.45], 'MarkerSize', 12);

set(findobj(ax, '-property', 'Clipping'), 'Clipping', 'off');
axis(ax, 'off');

ov = axes('Parent', fig, 'Position', get(ax, 'Position'), 'Color', 'none', 'XLim', [0 1], 'YLim', [0 1], ...
          'Visible', 'off', 'HitTest', 'off');
hold(ov, 'on');  H.ov = ov;
text(ov, 0.012, 0.975, 'SPRING-LEGGED JUMPING INSPECTION ROBOT', 'Color', [0.85 0.97 1], 'FontSize', 12, ...
     'FontWeight', 'bold', 'VerticalAlignment', 'top');
text(ov, 0.012, 0.935, 'DIGITAL TWIN V7  |  3D MISSION VIEW  |  COIL-SPRING LEGS  |  RUNNING JUMP', 'Color', cy, 'FontSize', 8.5, ...
     'FontName', fixedFont, 'VerticalAlignment', 'top');
H.ai    = text(ov, 0.012, 0.895, '', 'Color', [0.85 0.70 1], 'FontSize', 8.5, 'FontName', fixedFont, ...
               'VerticalAlignment', 'top', 'Interpreter', 'none');
H.route = text(ov, 0.012, 0.862, '', 'Color', [0.55 1 0.6], 'FontSize', 8.5, 'FontName', fixedFont, ...
               'VerticalAlignment', 'top', 'Interpreter', 'none');
H.team  = text(ov, 0.012, 0.829, '', 'Color', [1 0.85 0.4], 'FontSize', 8.5, 'FontName', fixedFont, ...
               'VerticalAlignment', 'top', 'Interpreter', 'none');
H.terr  = text(ov, 0.012, 0.796, '', 'Color', [0.95 0.80 0.55], 'FontSize', 8.5, 'FontName', fixedFont, ...
               'VerticalAlignment', 'top', 'Interpreter', 'none');
H.coll  = text(ov, 0.012, 0.763, '', 'Color', [0.6 1 0.6], 'FontSize', 8.5, 'FontName', fixedFont, ...
               'VerticalAlignment', 'top', 'Interpreter', 'none');
H.fault = text(ov, 0.5, 0.885, '', 'Color', [1 0.45 0.3], 'FontSize', 10, 'FontWeight', 'bold', ...
               'FontName', fixedFont, 'HorizontalAlignment', 'center', 'VerticalAlignment', 'top');
H.camTxt = text(ov, 0.988, 0.975, '', 'Color', [1 0.85 0.4], 'FontSize', 9, 'FontName', fixedFont, ...
                'HorizontalAlignment', 'right', 'VerticalAlignment', 'top');
H.banner = text(ov, 0.5, 0.965, '', 'HorizontalAlignment', 'center', 'VerticalAlignment', 'top', ...
                'FontSize', 11, 'FontWeight', 'bold', 'Color', 'w', 'BackgroundColor', [0.05 0.08 0.12], ...
                'EdgeColor', cy, 'Margin', 6, 'Interpreter', 'none');
H.bBat  = hudBar(ov, 0.115, 0.125, 0.16, 0.022, 'BATTERY', fixedFont);
H.bHlt  = hudBar(ov, 0.115, 0.085, 0.16, 0.022, 'SPRING HEALTH', fixedFont);
H.bSpr  = hudBar(ov, 0.115, 0.045, 0.16, 0.022, 'LEG CROUCH', fixedFont);
H.bLand = hudBar(ov, 0.115, 0.165, 0.16, 0.022, 'LANDING SAFE', fixedFont);
H.speed = text(ov, 0.985, 0.07, '', 'Color', [0.85 0.97 1], 'FontSize', 20, 'FontWeight', 'bold', ...
               'HorizontalAlignment', 'right', 'VerticalAlignment', 'middle');
H.mode  = text(ov, 0.985, 0.135, '', 'Color', cy, 'FontSize', 10, 'FontName', fixedFont, ...
               'HorizontalAlignment', 'right', 'VerticalAlignment', 'middle');
H.pause = text(ov, 0.5, 0.5, '', 'Color', [1 1 1], 'FontSize', 22, 'FontWeight', 'bold', ...
               'HorizontalAlignment', 'center');

dk = [0.06 0.08 0.11];
ao = axes('Parent', fig, 'Position', [0.045 0.055 0.64 0.14], 'Box', 'on', 'Color', dk, ...
          'XColor', [0.6 0.8 0.9], 'YColor', [0.6 0.8 0.9]);
hold(ao, 'on');  H.axOver = ao;
nz = numel(env.zone.name);
for q = 1:nz
    x0 = env.zone.x0(q);  x1 = env.zone.x1(q);
    if q == 1,  x0 = -0.5; end
    if q == nz, x1 = env.xEnd + 0.5; end
    xs = linspace(x0, x1, 80);
    patch(ao, [xs, fliplr(xs)], [terrainZ(env, xs), -0.2*ones(size(xs))], env.zone.color(q,:), 'EdgeColor', 'none');
    text(ao, 0.5*(x0 + x1), 0.62, env.zone.name{q}, 'HorizontalAlignment', 'center', 'FontSize', 7, 'Color', 'w');
end
for i = 1:size(env.obs,1)
    xo = env.obs(i,1);  wo = env.obs(i,2);  ho = env.obs(i,3);  zg = terrainZ(env, xo);
    patch(ao, [xo, xo + wo, xo + wo, xo], zg + [0 0 ho ho], [0.55 0.50 0.45], 'EdgeColor', 'k');
    text(ao, xo + wo/2, zg + ho + 0.07, sprintf('%d', i), 'HorizontalAlignment', 'center', 'FontSize', 8, 'Color', 'w');
end
if env.dynOn
    for i = 1:numel(env.dyn)
        mk = 'v';  if env.dyn(i).rock, mk = 'o'; end
        plot(ao, env.dyn(i).x, terrainZ(env, env.dyn(i).x) + 0.04, mk, 'MarkerFaceColor', [1 0.6 0.1], ...
             'MarkerEdgeColor', 'k', 'MarkerSize', 7);
        text(ao, env.dyn(i).x, terrainZ(env, env.dyn(i).x) + 0.14, sprintf('M%d', i), 'HorizontalAlignment', 'center', ...
             'FontSize', 7, 'Color', [1 0.8 0.4]);
    end
end
H.trail2 = plot(ao, NaN, NaN, '-', 'Color', cy, 'LineWidth', 1.5);
H.dot2   = plot(ao, NaN, NaN, 'o', 'MarkerFaceColor', [1 0.3 0.3], 'MarkerEdgeColor', 'w', 'MarkerSize', 7);
set(ao, 'XLim', [-0.5 env.xEnd + 0.5], 'YLim', [-0.05 0.75]);
xlabel(ao, 'x (m)');  title(ao, 'Mission overview', 'FontSize', 10, 'Color', [0.85 0.97 1]);

sx0 = 0.735;  sw = 0.255;  sh = 0.075;
yb  = [0.905 0.8075 0.71 0.6125 0.515];
ttl = {'LIDAR RANGE (m)', 'IMU GYRO (deg/s)', 'IMU ACCEL (g)', 'BATTERY SOC (%)', 'SPRING HEALTH (%)'};
yl  = {[0 1.0], [-200 200], [-1 6], [0 100], [60 105]};
cl  = {[0.2 0.9 1], [0.5 0.9 0.4], [1 0.85 0.3], [0.3 0.9 0.5], [0.8 0.6 1]};
for q = 1:5
    H.dash(q) = dashStrip(fig, [sx0 yb(q) sw sh], ttl{q}, yl{q}, cl{q}, fixedFont);
end

ai = axes('Parent', fig, 'Position', [0.70 0.055 0.295 0.43], 'Color', dk, 'Box', 'on', ...
          'XTick', [], 'YTick', [], 'XLim', [0 1], 'YLim', [0 1], 'XColor', [0.2 0.5 0.6], 'YColor', [0.2 0.5 0.6]);
H.axInfo = ai;
H.info = text(ai, 0.03, 0.985, '', 'VerticalAlignment', 'top', 'FontName', fixedFont, ...
              'FontSize', 8, 'Color', [0.78 0.95 1], 'Interpreter', 'none');
end

function B = dashStrip(fig, pos, ttl, ylim_, col, fnt)
B.ax = axes('Parent', fig, 'Position', pos, 'Color', [0.06 0.08 0.11], 'Box', 'on', 'XColor', [0.3 0.5 0.6], ...
            'YColor', [0.3 0.5 0.6], 'FontSize', 7, 'YLim', ylim_, 'XLim', [0 8], 'XTickLabel', []);
hold(B.ax, 'on');  grid(B.ax, 'on');
set(B.ax, 'GridColor', [0.3 0.5 0.6]);
B.l  = plot(B.ax, NaN, NaN, '-', 'Color', col, 'LineWidth', 1.1);
B.lf = plot(B.ax, NaN, NaN, '-', 'Color', [1 0.3 0.25], 'LineWidth', 1.8);
B.t  = text(B.ax, 0.01, 0.94, ttl, 'Units', 'normalized', 'Color', [0.7 0.9 1], 'FontSize', 7.5, ...
            'FontName', fnt, 'VerticalAlignment', 'top');
B.v  = text(B.ax, 0.99, 0.94, '', 'Units', 'normalized', 'Color', 'w', 'FontSize', 8, 'FontName', fnt, ...
            'HorizontalAlignment', 'right', 'VerticalAlignment', 'top');
end

function updateDashboard(H, S, k, t)
F = S.F;  D = S.D;
k0 = find(F(1:k,1) >= t - 8, 1);
if isempty(k0), k0 = k; end
r  = k0:k;  tt = F(r,1);  fm = F(r,19);
chans = {D(r,1), D(r,2), D(r,3), D(r,4), D(r,6)};
bits  = [1 8 8 2 4];
for q = 1:5
    yv = chans{q};  yf = yv;
    yf(bitand(fm, bits(q)) == 0) = NaN;
    set(H.dash(q).l,  'XData', tt, 'YData', yv);
    set(H.dash(q).lf, 'XData', tt, 'YData', yf);
    set(H.dash(q).ax, 'XLim', [t - 8, t + 0.01]);
end
if isnan(D(k,1)), s1 = 'NO ECHO'; else, s1 = sprintf('%.2f m', D(k,1)); end
set(H.dash(1).v, 'String', s1);
set(H.dash(2).v, 'String', sprintf('%.0f deg/s', D(k,2)));
set(H.dash(3).v, 'String', sprintf('%.2f g', D(k,3)));
set(H.dash(4).v, 'String', sprintf('%.0f %%   %.0f W', D(k,4), D(k,5)));
set(H.dash(5).v, 'String', sprintf('%.1f %%   %.0f N', D(k,6), D(k,7)));
end

function B = hudBar(ov, x0, y0, w, h, label, fnt)
B.x0 = x0;  B.w = w;  B.y0 = y0;  B.h = h;
B.bg   = patch(ov, [x0 x0+w x0+w x0], [y0 y0 y0+h y0+h], [0.06 0.09 0.13], 'EdgeColor', [0.3 0.6 0.7], 'FaceAlpha', 0.85);
B.fill = patch(ov, [x0 x0 x0 x0], [y0 y0 y0+h y0+h], [0.2 0.9 0.5], 'EdgeColor', 'none');
B.lab  = text(ov, x0 - 0.008, y0 + h/2, label, 'HorizontalAlignment', 'right', 'Color', [0.7 0.9 1], ...
              'FontSize', 8, 'FontName', fnt);
B.val  = text(ov, x0 + w + 0.008, y0 + h/2, '', 'HorizontalAlignment', 'left', 'Color', 'w', ...
              'FontSize', 8, 'FontName', fnt);
end

function setBar(B, f, col, str, vis)
f = min(max(f, 0), 1);
set(B.fill, 'XData', B.x0 + [0 B.w*f B.w*f 0], 'FaceColor', col);
set(B.val, 'String', str);
v = 'off';  if vis, v = 'on'; end
set([B.bg B.fill B.lab B.val], 'Visible', v);
end

function [Ms, Mg] = meshArch(Ms, Mg, xa, ~, yh, h)
Ms = meshBox(Ms, [xa,  yh, h/2], [0.07 0.07 h], [0.20 0.21 0.25]);
Ms = meshBox(Ms, [xa, -yh, h/2], [0.07 0.07 h], [0.20 0.21 0.25]);
nb = 14;  ys = linspace(-yh, yh, nb + 1);
for b = 1:nb
    if mod(b,2) == 1, col = [0.96 0.96 0.96]; else, col = [0.08 0.08 0.1]; end
    Ms = meshBox(Ms, [xa, 0.5*(ys(b) + ys(b+1)), h + 0.04], [0.09, ys(b+1) - ys(b), 0.08], col);
end
Mg = meshBox(Mg, [xa - 0.05, 0, h - 0.012], [0.01, 2*yh, 0.012], [0.25 0.9 1]);
end

function addSky(ax, sunDir)
R = 110;  c0 = [7.5 0 0];
[X, Y, Z] = sphere(72);
t = max(Z, 0).^0.55;
hor = [1.00 0.80 0.60];  mid = [0.62 0.78 0.96];  top = [0.10 0.27 0.62];
D = X*sunDir(1) + Y*sunDir(2) + Z*sunDir(3);
g = exp(-(acos(min(max(D, -1), 1))/0.45).^2);
C = zeros([size(X) 3]);
glow = [0.35 0.20 0.0];
for k = 1:3
    ck = hor(k) + (mid(k) - hor(k))*sstep(0, 0.4, t) + (top(k) - mid(k))*sstep(0.35, 1, t) + glow(k)*g;
    C(:,:,k) = min(max(ck, 0), 1);
end
surf(ax, c0(1) + R*X, c0(2) + R*Y, c0(3) + R*Z, C, 'EdgeColor', 'none', 'FaceColor', 'interp', 'FaceLighting', 'none');
sp = c0 + 100*sunDir;
[Xs, Ys, Zs] = sphere(20);
surf(ax, sp(1) + 5*Xs, sp(2) + 5*Ys, sp(3) + 5*Zs, 'FaceColor', [1 0.97 0.85], 'EdgeColor', 'none', 'FaceLighting', 'none');
surf(ax, sp(1) + 10*Xs, sp(2) + 10*Ys, sp(3) + 10*Zs, 'FaceColor', [1 0.9 0.6], 'EdgeColor', 'none', ...
     'FaceLighting', 'none', 'FaceAlpha', 0.25);
surf(ax, sp(1) + 20*Xs, sp(2) + 20*Ys, sp(3) + 20*Zs, 'FaceColor', [1 0.8 0.5], 'EdgeColor', 'none', ...
     'FaceLighting', 'none', 'FaceAlpha', 0.12);
end

function addTerrain(ax, env)
xv = unique([linspace(-45, -3, 40), linspace(-3, env.xEnd + 3, 640), linspace(env.xEnd + 3, env.xEnd + 62, 40), ...
             env.xk(:)', env.zone.x0, env.zone.x1(end)]);
yv = unique([linspace(-62, -4, 36), linspace(-4, -0.95, 24), linspace(-0.95, 0.95, 14), ...
             linspace(0.95, 4, 24), linspace(4, 62, 36)]);
[X, Y] = meshgrid(xv, yv);
Hh = hill3D(env, X, Y);
Z  = repmat(terrainZ(env, xv), numel(yv), 1) + Hh;
nx = numel(xv);  ny = numel(yv);

zc = zeros(nx, 3);  ga = zeros(1, nx);
grassAmt = [0.55 0.50 0.15 0.50 0.85 0.05  0.55 0.50 0.15 0.50 0.85 0.05];
for i = 1:nx
    zi = find(xv(i) >= env.zone.x0, 1, 'last');
    if isempty(zi), zi = 1; end
    zc(i,:) = env.zone.color(zi,:);  ga(i) = grassAmt(zi);
end
GA  = repmat(ga, ny, 1);
ay  = abs(Y);
nz1 = 0.5 + 0.25*sin(37*X + 11*Y) + 0.25*sin(23*X - 29*Y + 2);
nz2 = 0.5 + 0.5*sin(1.3*X).*sin(0.9*Y + 1);
centerMark = (ay < 0.012) & (mod(X, 0.5) < 0.25) & (X >= 0) & (X < 3.5);
edgeMark   = (abs(ay - 0.86) < 0.014) & (X >= 0) & (X < 3.5);
gcol = [0.20 0.38 0.14];  rockCol = [0.45 0.42 0.40];  snowCol = [0.95 0.96 1.0];
haze = [0.86 0.78 0.70];  edgeCol = [0.95 0.80 0.15];
rockA = sstep(3, 7, Hh);  snowA = sstep(8.5, 11, Hh);
e2 = sstep(0.9, 1.4, ay);
fog = 0.88*sstep(22, 78, hypot(X - 7.5, Y));
C = zeros(ny, nx, 3);
for k = 1:3
    ZCk = repmat(zc(:,k)', ny, 1);
    trk = ZCk.*(0.88 + 0.2*nz1);
    trk(centerMark) = 0.95;
    trk(edgeMark)   = edgeCol(k);
    off = GA.*(gcol(k)*(0.75 + 0.6*nz2) + 0.05*nz1) + (1 - GA).*ZCk*0.85.*(0.9 + 0.2*nz1);
    off = off.*(1 - rockA) + rockCol(k)*rockA;
    off = off.*(1 - snowA) + snowCol(k)*snowA;
    ck  = trk.*(1 - e2) + off.*e2;
    ck  = ck.*(1 - fog) + haze(k)*fog;
    C(:,:,k) = min(max(ck, 0), 1);
end
surf(ax, X, Y, Z, C, 'EdgeColor', 'none', 'FaceColor', 'interp', 'FaceLighting', 'gouraud', ...
     'AmbientStrength', 0.6, 'DiffuseStrength', 0.7, 'SpecularStrength', 0.04);
end

function R = buildRobot(ax, p, env)
%#ok<INUSD>
L = p.L;  W = p.W;  Hc = p.H;
Mb = meshNew();
Mb = meshBox(Mb, [0 0 0], [L W Hc], [0.12 0.38 0.80]);                         % torso
Mb = meshBox(Mb, [0 0 -Hc/2 - 0.004], [L*0.92 W*0.9 0.008], [0.10 0.10 0.12]);  % belly plate
Mb = meshBox(Mb, [L/2 + 0.02, 0, 0.005], [0.05 0.10 0.05], [0.15 0.15 0.17]);   % head / sensor pod
Mb = meshBox(Mb, [-L/2 - 0.006, 0, -0.01], [0.012 W*0.9 Hc*0.55], [0.15 0.15 0.17]);   % rear plate
Mb = meshBox(Mb, [-0.035 0 Hc/2 + 0.005], [0.15 0.17 0.01], [0.04 0.09 0.26]);  % solar skin
Mb = meshBox(Mb, [-0.13 0 Hc/2 + 0.035], [0.02 0.02 0.07], [0.85 0.85 0.88]);   % antenna
Mb = meshCyl(Mb, [-0.13 0 Hc/2 + 0.065], 0.028, 0.03, [0.15 0.15 0.18], 12);
Mb = meshQuad(Mb, [-0.13 + 0.028*[1 0 -1 0]', 0.028*[0 1 0 -1]', (Hc/2 + 0.095)*ones(4,1)], [0.2 0.2 0.24]);
Mb = meshBox(Mb, [0.08 0 Hc/2 + 0.012], [0.05 0.05 0.024], [0.15 0.15 0.18]);   % lidar base
% two tendon winch drums on the back (they wind the leg springs for the jump)
Mb = meshCyl(Mb, [-0.075,  0.060, Hc/2 + 0.010], 0.026, 0.030, [0.72 0.46 0.20], 14);
Mb = meshCyl(Mb, [-0.075, -0.060, Hc/2 + 0.010], 0.026, 0.030, [0.72 0.46 0.20], 14);
Mb = meshBox(Mb, [-0.075 0 Hc/2 + 0.020], [0.030 0.150 0.020], [0.18 0.19 0.22]);  % winch cross-frame
% outrigger arms + guide towers: the four spring legs hang from these
for hx = [-p.wb/2 p.wb/2]
    for hy = [-1 1]
        Mb = meshBox(Mb, [hx, hy*(W/2 + (p.wy - W/2)/2), -Hc/2 + 0.012], [0.04, p.wy - W/2 + 0.03, 0.014], [0.20 0.21 0.25]);
        Mb = meshCyl(Mb, [hx, hy*p.wy, -Hc/2 - 0.002], 0.024, 0.05, [0.18 0.19 0.22], 14);
    end
end
R.bodyV = Mb.V;
R.bodyH = meshPatch(ax, Mb, 'FaceLighting', 'flat', 'AmbientStrength', 0.55, 'DiffuseStrength', 0.85, ...
                    'SpecularStrength', 0.35, 'SpecularExponent', 20);
Mg = meshNew();
Mg = meshBox(Mg, [0 0 0.012], [L - 0.02, W + 0.003, 0.010], [0.10 0.90 1.0]);
Mg = meshBox(Mg, [L/2 + 0.047,  0.03, 0.012], [0.006 0.03 0.015], [1 1 0.85]);   % eyes
Mg = meshBox(Mg, [L/2 + 0.047, -0.03, 0.012], [0.006 0.03 0.015], [1 1 0.85]);
Mg = meshBox(Mg, [-L/2 - 0.013,  0.07, 0.005], [0.006 0.03 0.015], [1 0.1 0.1]);
Mg = meshBox(Mg, [-L/2 - 0.013, -0.07, 0.005], [0.006 0.03 0.015], [1 0.1 0.1]);
Mg = meshBox(Mg, [L/2 + 0.03, 0, 0.036], [0.016 0.05 0.016], [1 0.1 0.1]);       % lidar strip
R.glowV = Mg.V;
R.glowH = meshPatch(ax, Mg, 'FaceLighting', 'none');

[V, Fc, Cc] = wheelMesh(0.04, 0.026, 24, [0.04 0.84 1.0], [0.9 0.92 0.95], [0.04 0.84 1.0], [0.08 0.1 0.14]);
R.rwV = V;  R.rwOff = [0.08 0 Hc/2 + 0.036];
R.rwH = patch('Parent', ax, 'Vertices', V, 'Faces', Fc, 'FaceVertexCData', Cc, 'FaceColor', 'flat', ...
              'EdgeColor', 'none', 'FaceLighting', 'none');

% ---- the four telescopic spring legs: guide sleeve, plunger rod, coil spring, collar, foot pad ----
for q = 1:4
    R.lgU(q) = plot3(ax, NaN, NaN, NaN, '-', 'Color', [0.22 0.24 0.29], 'LineWidth', 7);       % guide sleeve
    R.lgL(q) = plot3(ax, NaN, NaN, NaN, '-', 'Color', [0.88 0.90 0.94], 'LineWidth', 3);       % plunger rod
    R.lgC(q) = plot3(ax, NaN, NaN, NaN, '-', 'Color', [1.00 0.55 0.12], 'LineWidth', 1.8);     % coil spring
    R.kn(q)  = plot3(ax, NaN, NaN, NaN, 'o', 'MarkerSize', 5, 'MarkerFaceColor', [0.1 0.9 1], 'MarkerEdgeColor', 'k');
    R.pd(q)  = plot3(ax, NaN, NaN, NaN, 'o', 'MarkerSize', 9, 'MarkerFaceColor', [0.12 0.12 0.14], 'MarkerEdgeColor', [0.9 0.9 0.9]);
end

zr = Hc/2 + 0.0105;  gl = [];
for xx = [-0.085 -0.035 0.015]
    gl = [gl; xx -0.085 zr; xx 0.085 zr; NaN NaN NaN]; %#ok<AGROW>
end
for yy = [-0.055 0 0.055]
    gl = [gl; -0.11 yy zr; 0.04 yy zr; NaN NaN NaN]; %#ok<AGROW>
end
R.gridL = gl;
R.gridH = plot3(ax, NaN, NaN, NaN, '-', 'Color', [0.35 0.55 0.95], 'LineWidth', 0.8);

R.beamV = [L/2 + 0.05  0.07 0.0; L/2 + 0.05 -0.07 0.0; 0.80 -0.30 -0.115; 0.80 0.30 -0.115];
R.beamH = patch('Parent', ax, 'Vertices', R.beamV, 'Faces', [1 2 3 4], 'FaceColor', [1 0.95 0.7], ...
                'FaceAlpha', 0.10, 'EdgeColor', 'none', 'FaceLighting', 'none');

R.footF   = boxFaces();
R.arm     = plot3(ax, NaN, NaN, NaN, '-', 'Color', [1 0.6 0.1], 'LineWidth', 3);     % dorsal flip arm
R.armFoot = patch('Parent', ax, 'Vertices', boxVerts([0 0 0.1], [0.075 0.075 0.008]), 'Faces', R.footF, ...
                  'FaceColor', [0.12 0.12 0.14], 'EdgeColor', 'none', 'FaceLighting', 'flat');

R.shadow = patch('Parent', ax, 'Vertices', zeros(24,3), 'Faces', 1:24, 'FaceColor', [0 0 0], ...
                 'FaceAlpha', 0.3, 'EdgeColor', 'none', 'FaceLighting', 'none');

R.plan  = plot3(ax, NaN, NaN, NaN, '--', 'Color', [1 0.35 0.95], 'LineWidth', 2);
R.ring  = plot3(ax, NaN, NaN, NaN, '-', 'Color', [0.3 1 0.5], 'LineWidth', 2.5);
R.ghostF = zeros(48, 4);
for j = 1:8, R.ghostF(6*(j-1) + (1:6), :) = boxFaces() + 8*(j-1); end
R.ghost = patch('Parent', ax, 'Vertices', zeros(64,3), 'Faces', R.ghostF, 'FaceColor', [0.9 0.4 1], ...
                'FaceAlpha', 0.10, 'EdgeColor', [0.9 0.35 1], 'LineStyle', ':', 'FaceLighting', 'none');

nxq = 45;  nyq = 9;
R.scanF = zeros(6, 4);
for i = 1:6, R.scanF(i,:) = [1, i + 1, i + 2, 1]; end
R.scanFan = patch('Parent', ax, 'Vertices', zeros(8,3), 'Faces', R.scanF, 'FaceColor', [0.2 0.9 1], ...
                  'FaceAlpha', 0.10, 'EdgeColor', 'none', 'FaceLighting', 'none');
R.scanPts = plot3(ax, NaN, NaN, NaN, '.', 'Color', [0.15 0.95 0.65], 'MarkerSize', 5);
R.scanHit = plot3(ax, NaN, NaN, NaN, '.', 'Color', [1 0.25 0.2], 'MarkerSize', 9);
R.pulse   = plot3(ax, NaN, NaN, NaN, '-', 'Color', [0.4 1 1], 'LineWidth', 2);
R.beam    = plot3(ax, NaN, NaN, NaN, '-', 'Color', [1 0.2 0.2], 'LineWidth', 1.4);
R.trail   = plot3(ax, NaN, NaN, NaN, '-', 'Color', [0.2 0.7 1], 'LineWidth', 2);
R.nq = [nxq nyq];
end

%% =====================================================================
%%  3D FRAME UPDATE
%% =====================================================================
function drawFrame3D(H, p, env, S, k, ctl, finished, wNow)
row = S.F(k,:);
t = row(1); x = row(2); z = row(3); phi = row(4);
comp = row(6); leg = row(7); mode = row(8); msgId = row(9);
E = row(10); Fs = row(12); vx = row(13); act = row(15); hlt = row(16);
y = row(17); yaw = row(18); fm = row(19);
curPose([y yaw]);
envY = envAtY(env, y);
rho = min(comp/p.maxTravel, 1);
R = H.R;

% ---- body: bounces on the leg springs while walking, drops while the springs are charged for a jump ----
zb = z + bobZ(p, row);
if mode == 3 || mode == 4, zb = z - comp; end
setV(R.bodyH, R.bodyV, x, zb, phi);
setV(R.glowH, R.glowV, x, zb, phi);
aRW = (2 + 14*(mode == 5))*wNow;
setV(R.rwH, bsxfun(@plus, rotY(R.rwV, aRW), R.rwOff), x, zb, phi);
[X, Y, Z] = b2w(R.gridL, x, zb, phi); set(R.gridH, 'XData', X, 'YData', Y, 'ZData', Z);
setV(R.beamH, R.beamV, x, zb, phi);
set(R.beamH, 'Visible', ternary(mode == 7 || mode == 8, 'off', 'on'));

% ---- four telescopic spring legs (coil compresses in the crouch, stretches at launch, retracts in flight) ----
[Hw, Kw, Fw, Cw] = legGeometry(p, env, row);
setLegs(R.lgU, R.lgL, R.lgC, R.kn, R.pd, Hw, Kw, Fw, Cw);

% ---- dorsal flip arm (self-righting only) ----
if mode == 8
    nT = 5 + round(8*rho);  amp = 0.012 + 0.008*rho;
    tt = linspace(0, 1, 18*nT + 1)';  ang = 2*pi*nT*tt;
    scol = [0.2 + 0.8*rho, 0.90 - 0.60*rho, 0.30*(1 - rho) + 0.1];
    armLen = 0.03 + leg;
    Lh = [p.armSR + amp*cos(ang), amp*sin(ang), p.H/2 + armLen*tt];
    [X, Y, Z] = b2w(Lh, x, z, phi);
    set(R.arm, 'XData', X, 'YData', Y, 'ZData', Z, 'Color', scol);
    setV(R.armFoot, boxVerts([p.armSR 0 p.H/2 + armLen + 0.004], [0.075 0.075 0.008]), x, z, phi);
    set(R.armFoot, 'Visible', 'on');
else
    set(R.arm, 'XData', NaN, 'YData', NaN, 'ZData', NaN);
    set(R.armFoot, 'Visible', 'off');
end

% ---- soft ground shadow ----
ag = linspace(0, 2*pi, 25)';  ag(end) = [];
xv = x + 0.17*cos(ag);  yv = y + 0.12*sin(ag);
zv = groundHeight(envY, xv) + 0.005;
hAbove = max(z - groundHeight(envY, x) - p.restZ, 0);
set(R.shadow, 'Vertices', [xv yv zv], 'FaceAlpha', 0.38/(1 + 5*hAbove));

% ---- LiDAR scan layer ----
if mode == 1 || mode == 2 || mode == 10 || mode == 11
    fsNow = faultState(env, x);
    [sx, sy, sz] = b2w([p.L/2 + 0.05 0 0.012], x, z, phi);
    xs = sx;  Rg = p.senseRange*fsNow.lRange;
    xq = linspace(xs, xs + Rg, R.nq(1));  yq = y + linspace(-0.42, 0.42, R.nq(2));
    [XQ, YQ] = meshgrid(xq, yq);
    ZQ = groundHeightXY(env, XQ, YQ) + 0.003 + 0.002*sin(900*XQ + 40*t);
    hit = (ZQ - terrainZ(env, XQ)) > 0.01;
    set(R.scanPts, 'XData', XQ(~hit), 'YData', YQ(~hit), 'ZData', ZQ(~hit));
    set(R.scanHit, 'XData', XQ(hit),  'YData', YQ(hit),  'ZData', ZQ(hit));
    yf = y + linspace(-0.42, 0.42, 7);  xf = (xs + Rg)*ones(1,7);
    zf = groundHeightXY(env, xf, yf) + 0.004;
    set(R.scanFan, 'Vertices', [sx sy sz; xf' yf' zf']);
    xp = xs + mod(wNow*1.3, Rg);  yp = y + linspace(-0.42, 0.42, 9);
    set(R.pulse, 'XData', xp*ones(1,9), 'YData', yp, 'ZData', groundHeightXY(env, xp*ones(1,9), yp) + 0.005);
    q = find(env.obs(:,1) > xs & abs(y) < env.obsHalfW(:) + 0.1, 1);
    if ~isempty(q) && (env.obs(q,1) - xs) <= Rg
        set(R.beam, 'XData', [sx env.obs(q,1)], 'YData', [sy y], ...
            'ZData', [sz terrainZ(env, env.obs(q,1)) + env.obs(q,3)], 'Color', [1 0.2 0.2], 'LineStyle', '-');
    else
        set(R.beam, 'XData', [sx xs + Rg], 'YData', [sy y], 'ZData', [sz terrainZ(env, xs + Rg)], ...
            'Color', [0.2 1 0.5], 'LineStyle', '--');
    end
    set(R.scanFan, 'Visible', 'on');
else
    set([R.scanPts R.scanHit R.pulse R.beam], 'XData', NaN, 'YData', NaN, 'ZData', NaN);
    set(R.scanFan, 'Visible', 'off');
end

% ---- planned jump: arc, ghosts, landing target ----
if act > 0 && ~isempty(S.ev(act).arcX)
    pl = S.ev(act).plan;
    set(R.plan, 'XData', S.ev(act).arcX, 'YData', y*ones(size(S.ev(act).arcX)), 'ZData', S.ev(act).arcZ);
    gi = unique(round(linspace(1, pl.fl.n, 8)));
    GV = zeros(64, 3);
    for j = 1:8
        jj = gi(min(j, numel(gi)));
        [GX, GY, GZ] = b2w(boxVerts([0 0 0], [p.L p.W p.H]), pl.xLaunch + pl.fl.x(jj), pl.fl.z(jj), pl.fl.phi(jj));
        GV(8*(j-1) + (1:8), :) = [GX GY GZ];
    end
    set(R.ghost, 'Vertices', GV, 'Visible', 'on');
    ar = linspace(0, 2*pi, 40);
    xr = pl.xLaunch + pl.fl.xTD + 0.14*cos(ar);  yr = y + 0.14*sin(ar);
    set(R.ring, 'XData', xr, 'YData', yr, 'ZData', groundHeight(envY, xr) + 0.006);
else
    set(R.plan, 'XData', NaN, 'YData', NaN, 'ZData', NaN);
    set(R.ring, 'XData', NaN, 'YData', NaN, 'ZData', NaN);
    set(R.ghost, 'Visible', 'off');
end

set(R.trail, 'XData', S.F(1:k,2), 'YData', S.F(1:k,17), 'ZData', S.F(1:k,3));

% ---- dust burst on landing ----
iN = find(S.F(1:k,8) ~= 6, 1, 'last');
if mode == 6 && ~isempty(iN)
    tl = t - S.F(iN,1);
else
    tl = Inf;
end
if tl < 0.5
    D = H.dustDir;  sp = 0.5 + 0.9*D(:,4);
    zg0 = groundHeight(envY, x);
    set(H.dust, 'XData', x + D(:,1).*sp*tl, 'YData', y + D(:,2).*sp*tl*0.8, ...
        'ZData', zg0 + 0.01 + max(D(:,3).*sp*tl*1.2 - 2.0*tl^2, 0), 'MarkerSize', max(2, 14*(1 - tl/0.5)));
else
    set(H.dust, 'XData', NaN, 'YData', NaN, 'ZData', NaN);
end

% ---- moving obstacles + radar track estimate ----
if env.dynOn
    for i = 1:numel(env.dyn)
        d = env.dyn(i);  yd = dynY(d, S.dyn(i).t0, t);
        Vb = H.dynV{i};
        if d.rock
            roll = yd/d.hw;
            zd   = terrainZ(env, d.x) + d.hw + 0.025*abs(sin(2*roll));
            Vb   = rotX(Vb, roll);
            set(H.dynH(i), 'Vertices', Vb + repmat([d.x yd zd], size(Vb, 1), 1));
        else
            zd = terrainZ(env, d.x) + 0.06;
            set(H.dynH(i), 'Vertices', Vb + repmat([d.x yd zd], size(Vb, 1), 1));
            cB = [1 0.15 0.1];  if mod(wNow, 0.6) < 0.3, cB = [1 0.9 0.2]; end
            Cd = get(H.dynH(i), 'FaceVertexCData');  Cd(7:12, :) = repmat(cB, 6, 1);
            set(H.dynH(i), 'FaceVertexCData', Cd);
        end
        Ld = S.dyn(i).log;
        q = [];
        if ~isempty(Ld), q = find(Ld(:,1) <= t, 1, 'last'); end
        if ~isempty(q) && (t - Ld(q,1)) < 0.05 && Ld(q,5) ~= 3
            set(H.dynEst(i), 'XData', d.x, 'YData', Ld(q,3), 'ZData', zd + 0.14);
        else
            set(H.dynEst(i), 'XData', NaN, 'YData', NaN, 'ZData', NaN);
        end
    end
end

nTb = size(H.hub, 1);  BV = zeros(12*nTb, 3);
for j = 1:nTb
    a0 = wNow*(1.0 + 0.15*j) + j;
    for b = 0:2
        ab = a0 + b*2*pi/3;
        dR = [cos(ab) 0 sin(ab)];  dT = [-sin(ab) 0 cos(ab)];
        root = H.hub(j,:) + 0.12*dR;  tip = H.hub(j,:) + 2.3*dR;
        BV(12*(j-1) + 4*b + (1:4), :) = [root + 0.07*dT; root - 0.07*dT; tip - 0.015*dT; tip + 0.015*dT];
    end
end
set(H.blade, 'Vertices', BV);
V = H.cloudV0;  V(:,1) = V(:,1) + 0.8*wNow;
set(H.cloud, 'Vertices', V);

updateCamera(H, x, y, z, t, ctl.cam);

drawFollowers(H, p, env, S, t);

set(H.trail2, 'XData', S.F(1:k,2), 'YData', S.F(1:k,3));
set(H.dot2, 'XData', x, 'YData', z);

updateDashboard(H, S, k, t);

modes = {'WALKING (spring legs)','SENSING / PLANNING','CROUCHING ON THE MOVE (springs charging)','LEG EXTENSION (launch)','FLIGHT (legs retracted)', ...
         'LANDING','TUMBLING','SELF-RIGHTING','MISSION END','DETOUR (route)','YIELDING (traffic)'};
zi = zoneAt(env, x);
nDone = sum(~isnan([S.ev.tDone]) & ([S.ev.tDone] <= t));
soc = batterySOC(p, t, E);
cot = E/(p.m*p.g*max(x - 0.35, 0.5));
camNames = {'CHASE CAM','SIDE PROFILE','TOP-DOWN','CINEMATIC ORBIT','FRONT HERO SHOT'};
set(H.camTxt, 'String', sprintf('[C] %s', camNames{ctl.cam}));
set(H.mode, 'String', modes{mode});
set(H.speed, 'String', sprintf('%.2f m/s', vx));
if soc > 0.4, bc = [0.2 0.9 0.5]; elseif soc > 0.25, bc = [1 0.8 0.2]; else, bc = [1 0.3 0.3]; end
setBar(H.bBat, soc, bc, sprintf('%3.0f %%', 100*soc), true);
if hlt > 0.85, hc = [0.2 0.9 0.5]; elseif hlt > p.healthMin, hc = [1 0.8 0.2]; else, hc = [1 0.3 0.3]; end
setBar(H.bHlt, (hlt - 0.6)/0.4, hc, sprintf('%4.1f %%', 100*hlt), true);
setBar(H.bSpr, rho, [1 0.35 + 0.5*(1 - rho) 0.2], sprintf('%4.1f mm', 1000*comp), true);
qq = [];
if act > 0, qq = find(S.MC.idx == act, 1); end
if ~isempty(qq)
    pr = S.MC.prob(qq);
    setBar(H.bLand, pr, [0.3 0.8 1.0], sprintf('%5.1f %%', 100*pr), true);
else
    setBar(H.bLand, 0, [0.3 0.8 1.0], '', false);
end
if ctl.paused, set(H.pause, 'String', 'PAUSED'); else, set(H.pause, 'String', ''); end

fn = {'LIDAR', 'LEG MOTORS', 'LEG SPRING', 'IMU'};  fb = [1 2 4 8];
onF = fn(bitand(fm, fb) > 0);
if isempty(onF), set(H.fault, 'String', ''); else, set(H.fault, 'String', ['FAULT ACTIVE: ' strjoin(onF, ' + ')]); end
qa = find([S.ev.tSense] <= t, 1, 'last');
if ~env.aiOn
    set(H.ai, 'String', 'AI OFF - physics decision');
elseif isempty(qa) || S.ev(qa).cAI == 0
    set(H.ai, 'String', 'AI: scanning ahead ...');
else
    ea = S.ev(qa);
    set(H.ai, 'String', sprintf('AI #%d %s: %s %.0f%% -> %s%s', qa, env.obsName{qa}, aiName(ea.cAI), 100*ea.pAI, ...
            ea.decision, ternary(ea.override, '  (supervisor override)', '')));
end
ri = find(S.routeT <= t, 1, 'last');
if isempty(ri), ri = 1; end
rt = S.routes{ri};
set(H.route, 'String', sprintf('ROUTE %s: %.1f m ahead, %d re-plan(s)', upper(env.routeMode), rt.len, ri - 1));
set(H.routeC, 'XData', rt.x, 'YData', rt.y, 'ZData', terrainZ(env, rt.x) + 0.022);

tn = {'Asphalt','Gravel','Grass','Sand'};
if ~isempty(S.T)
    set(H.terr, 'String', sprintf('TERRAIN AI: %s (%.0f%%)  [true: %s]', tn{S.T(k,2)}, 100*S.T(k,3), tn{S.T(k,1)}));
else
    set(H.terr, 'String', '');
end
if ~isempty(S.C)
    lv = S.C(k,2);  ttcV = S.C(k,1);
    lblTxt = {'CLEAR', 'THREAT TRACKED (avoidance active)', 'COLLISION PREDICTED', 'COLLISION IMMINENT'};
    lblCol = [0.6 1 0.6; 1 0.85 0.3; 1 0.5 0.2; 1 0.2 0.2];
    cstr = ['COLLISION: ' lblTxt{lv+1}];
    if lv > 0, cstr = sprintf('%s  TTC %.2f s', cstr, ttcV); end
    set(H.coll, 'String', cstr, 'Color', lblCol(lv+1,:));
    if lv >= 2
        xe = x + p.vCruise*min(ttcV, 2);
        set(H.cpLine, 'XData', [x xe], 'YData', [y y], 'ZData', [z z] + 0.03);
    else
        set(H.cpLine, 'XData', NaN, 'YData', NaN, 'ZData', NaN);
    end
else
    set(H.coll, 'String', '');
end

L = {};
L{end+1} = sprintf('MODE      : %s', modes{mode});
L{end+1} = sprintf('TIME      : %6.2f s', t);
L{end+1} = sprintf('POSITION  : x %5.2f m  y %+5.2f m', x, y);
L{end+1} = sprintf('SPEED     : %4.2f m/s', vx);
L{end+1} = sprintf('TERRAIN   : %s (Crr %.2f)', env.zone.name{zi}, env.zone.crr(zi));
L{end+1} = sprintf('PITCH     : %6.1f deg  YAW %+5.1f deg', phi*180/pi, yaw*180/pi);
L{end+1} = sprintf('LEG SPRING: %5.1f mm   %5.0f N', 1000*comp, Fs);
L{end+1} = sprintf('ENERGY    : %6.2f J used   COT %.2f', E, cot);
L{end+1} = sprintf('OBSTACLES : %d of %d passed', nDone, size(env.obs,1));
if soc < 0.25, L{end+1} = '!! LOW BATTERY - return to base soon'; end
L{end+1} = repmat('-', 1, 46);
L{end+1} = 'DIGITAL-TWIN LOG (latest)';
if msgId > 0
    for m = max(1, msgId - 3):msgId
        ml = S.msgs{m};
        for j = 1:numel(ml)
            s = ml{j};
            if j == 1, s = ['> ' s]; else, s = ['  ' s]; end %#ok<AGROW>
            L{end+1} = s(1:min(end, 50)); %#ok<AGROW>
        end
    end
end
L{end+1} = repmat('-', 1, 46);
if finished, L{end+1} = 'FINISHED: press R to replay, Q to close'; end
L{end+1} = 'KEYS: SPACE pause | S slow | R replay | C cam | Q quit';
set(H.info, 'String', strjoin(L, char(10)));

if msgId > 0
    head = S.msgs{msgId}{1};
    set(H.banner, 'String', head, 'EdgeColor', bannerColor(head));
end
end

function updateCamera(H, x, y, z, t, camMode)
fig = H.fig;  ax = H.ax;
up = [0 0 1];
tgt = [x + 0.15, y, z + 0.02];
switch camMode
    case 1
        off = [-0.95 -1.05 0.50];  va = 38;
    case 2
        off = [0.10 -1.90 0.30];   va = 34;  tgt = [x + 0.12 y z + 0.03];
    case 3
        off = [0 0 2.4];           va = 40;  tgt = [x + 0.30 y z];  up = [0 1 0];
    case 4
        a = -pi/2 + 0.25*t;  r = 1.45 + 0.35*sin(0.25*t);
        off = [r*cos(a), r*sin(a), 0.40 + 0.2*sin(0.3*t)];  va = 38;
    otherwise
        off = [1.10 -0.60 0.14];   va = 42;  tgt = [x y z];
end
posN = tgt + off;
posN(3) = max(posN(3), 0.06);
st = getappdata(fig, 'cam');
if isempty(st) || st.mode ~= camMode
    pos = posN;  tg = tgt;
else
    pos = st.pos + 0.18*(posN - st.pos);
    tg  = st.tgt + 0.35*(tgt - st.tgt);
end
setappdata(fig, 'cam', struct('pos', pos, 'tgt', tg, 'mode', camMode));
set(ax, 'CameraPosition', pos, 'CameraTarget', tg, 'CameraUpVector', up, 'CameraViewAngle', va);
end

function c = bannerColor(s)
c = [0.25 0.90 1.00];
if strncmp(s,'EVENT',5) || strncmp(s,'TUMBLE',6) || strncmp(s,'RECOVERY WARN',13)
    c = [1.0 0.35 0.35];
elseif strncmp(s,'FAULT',5)
    c = [1.0 0.50 0.15];
elseif strncmp(s,'AI',2) || strncmp(s,'SUPERVISOR',10)
    c = [0.85 0.65 1.0];
elseif strncmp(s,'ADAPTIVE',8)
    c = [1.0 0.60 0.85];
elseif strncmp(s,'DYNAMIC',7) || strncmp(s,'YIELD',5)
    c = [1.0 0.75 0.25];
elseif strncmp(s,'ROUTE',5)
    c = [0.55 1.0 0.60];
elseif strncmp(s,'SENSE',5)
    c = [0.40 0.80 1.0];
elseif strncmp(s,'DECIDE',6) || strncmp(s,'PLAN',4) || strncmp(s,'CHECK',5)
    c = [1.0 0.90 0.30];
elseif strncmp(s,'LANDING',7) || strncmp(s,'RECOVERY',8) || strncmp(s,'MISSION',7) || strncmp(s,'HEALTH',6) || strncmp(s,'CLEAR',5)
    c = [0.40 1.0 0.50];
elseif strncmp(s,'SELF',4)
    c = [1.0 0.65 0.20];
elseif strncmp(s,'ACTION',6) || strncmp(s,'FLIGHT',6)
    c = [0.75 0.60 1.0];
end
end

%% =====================================================================
%%  ANALYTICS
%% =====================================================================
function f2 = showAnalytics(p, env, S)
F = S.F;  t = F(:,1);
f2 = figure('Name','Digital Twin - Analytics','NumberTitle','off','Position',[70 50 1320 780],'Color','w');

modeNames = {'Walking','Planning','Crouching','Launch','Flight','Landing','Tumble','Self-right','End', ...
             'Detour','Yielding'};
modeCol = [0.2 0.6 0.2; 0.9 0.7 0.1; 0.8 0.3 0.8; 0.9 0.2 0.2; 0.1 0.4 0.9; ...
           0.1 0.7 0.7; 0.5 0.1 0.1; 0.9 0.5 0.0; 0.4 0.4 0.4; 0.5 0.8 0.2; 1.0 0.4 0.6];

subplot(2,3,1); hold on; box on; grid on;
xx = linspace(0, env.xEnd + 0.5, 1800);
plot(xx, groundHeight(env, xx), 'k-', 'LineWidth', 1.4, 'HandleVisibility', 'off');
for m = 1:numel(modeNames)
    s = F(:,8) == m;
    if any(s)
        plot(F(s,2), F(s,3), '.', 'Color', modeCol(m,:), 'MarkerSize', 7, 'DisplayName', modeNames{m});
    end
end
xlabel('x (m)'); ylabel('COM height z (m)'); title('Trajectory by locomotion mode');
legend('Location', 'northwest', 'FontSize', 7);

subplot(2,3,2); hold on; box on; grid on;
plot(t, F(:,10), 'b-', 'LineWidth', 1.6);
xlabel('time (s)'); ylabel('cumulative energy (J)'); title('Energy consumption (gait + spring charging)');

subplot(2,3,3); hold on; box on; grid on;
yyaxis left;
plot(t, 1000*F(:,6), '-', 'LineWidth', 1.5);
plot([t(1) t(end)], 1000*p.maxTravel*[1 1], '--', 'Color', [0.5 0.5 0.5]);
ylabel('leg crouch (mm)');
yyaxis right;
plot(t, F(:,12), '-', 'LineWidth', 1.5);
ylabel('spring force (N)');
xlabel('time (s)'); title('Adaptive leg crouch and spring force');

subplot(2,3,4); hold on; box on; grid on;
plot(t, F(:,4)*180/pi, 'm-', 'LineWidth', 1.4);
plot([t(1) t(end)],  p.phiLimit*180/pi*[1 1], 'r--');
plot([t(1) t(end)], -p.phiLimit*180/pi*[1 1], 'r--');
xlabel('time (s)'); ylabel('pitch (deg)'); title('Body pitch (dashed = landing limit; -180 = inverted)');

subplot(2,3,5); hold on; box on; grid on;
n = numel(S.ev);
Er = [S.ev.Eroll];  Er(isinf(Er)) = NaN;
Ej = [S.ev.Ejump];  Ea = [S.ev.Eavoid];
bar(1:n, [Er(:), Ej(:), Ea(:)]);
legend('Step', 'Jump', 'Avoid', 'Location', 'northwest');
yTop = max([Er(:); Ej(:); Ea(:)]);
if isempty(yTop) || isnan(yTop), yTop = 1; end
for i = 1:n
    yv = max([Er(i), Ej(i), Ea(i), 0]);
    text(i, yv + 0.04*yTop, S.ev(i).decision, 'HorizontalAlignment', 'center', 'FontWeight', 'bold', 'Rotation', 90);
end
set(gca, 'XTick', 1:n);
xlabel('obstacle #'); ylabel('energy (J)'); title('Option energies and final decision (missing bar = not possible)');
ylim([0, 1.3*yTop]);

subplot(2,3,6); hold on; box on; grid on;
jumped = find([S.ev.executed]);
if ~isempty(jumped)
    pu = zeros(numel(jumped),1);  pc = pu;
    for q = 1:numel(jumped)
        pu(q) = S.ev(jumped(q)).plan.phiU;
        pc(q) = S.ev(jumped(q)).plan.phiC;
    end
    bar(1:numel(jumped), [pu, pc]);
    plot([0.4 numel(jumped) + 0.6],  p.phiLimit*180/pi*[1 1], 'r--');
    plot([0.4 numel(jumped) + 0.6], -p.phiLimit*180/pi*[1 1], 'r--');
    set(gca, 'XTick', 1:numel(jumped), 'XTickLabel', cellstr(num2str(jumped(:))));
    legend('no attitude control', 'reaction-wheel PD', 'limit', 'Location', 'northwest');
end
xlabel('jumped obstacle #'); ylabel('touchdown pitch (deg)'); title('Landing stability (digital-twin prediction)');

if exist('sgtitle', 'file')
    sgtitle('Adaptive Spring-Legged Jumping Inspection Robot - Digital Twin V7 analytics');
end
figure(f2);
end

%% =====================================================================
%%  ADVANCED ANALYTICS
%% =====================================================================
function f3 = showAdvanced(p, env, S, CM) %#ok<INUSL>
F = S.F;  t = F(:,1);
f3 = figure('Name','Digital Twin - Advanced analytics','NumberTitle','off', ...
            'Position',[100 70 1320 780],'Color','w');

subplot(2,3,1); hold on; box on; grid on;
soc = batterySOC(p, t, F(:,10));
yyaxis left;
plot(t, 100*soc, 'LineWidth', 1.6);
ylabel('battery SOC (%)');  ylim([0 105]);
yyaxis right;
plot(t, 100*F(:,16), 'LineWidth', 1.6);
ylabel('spring health (%)');  ylim([60 102]);
xlabel('time (s)'); title('Battery (with solar harvest) and spring health');

subplot(2,3,2); hold on; box on; grid on;
MC = S.MC;
if ~isempty(MC.idx)
    bar(1:numel(MC.idx), 100*MC.prob);
    for q = 1:numel(MC.idx)
        text(q, 100*MC.prob(q) + 2, sprintf('%.1f%%', 100*MC.prob(q)), 'HorizontalAlignment', 'center');
    end
    set(gca, 'XTick', 1:numel(MC.idx), 'XTickLabel', cellstr(num2str(MC.idx(:))));
    ylim([0 110]);
end
xlabel('jumped obstacle #'); ylabel('landing success (%)');
title(sprintf('Monte-Carlo landing risk (%d trials per jump)', MC.n));

subplot(2,3,3); hold on; box on; grid on;
cols = lines(max(numel(MC.idx), 1));
lg = {};
for q = 1:numel(MC.idx)
    plot(1000*MC.margin{q}, MC.phi{q}, '.', 'Color', cols(q,:), 'MarkerSize', 9);
    lg{end+1} = sprintf('obstacle #%d', MC.idx(q)); %#ok<AGROW>
end
plot([0 0], [-45 45], 'r--');
plot(xlim, p.phiLimit*180/pi*[1 1], 'r:');
plot(xlim, -p.phiLimit*180/pi*[1 1], 'r:');
xlabel('hind-foot landing margin past obstacle (mm)'); ylabel('touchdown pitch (deg)');
title('Landing dispersion (left of red line = hit obstacle)');
if ~isempty(lg), legend(lg, 'Location', 'best', 'FontSize', 7); end

subplot(2,3,4); hold on; box on;
if ~isempty(CM)
    imagesc(CM.mode);
    colormap(gca, [0.45 0.80 0.45; 0.98 0.82 0.25; 0.90 0.35 0.35]);
    caxis([1 3]);
    set(gca, 'YDir', 'normal', 'XTick', 1:numel(CM.w), 'XTickLabel', arrayfun(@(a) sprintf('%.0f', 1000*a), CM.w, 'UniformOutput', false), ...
        'YTick', 1:numel(CM.h), 'YTickLabel', arrayfun(@(a) sprintf('%.0f', 1000*a), CM.h, 'UniformOutput', false));
    for a = 1:numel(CM.h)
        for b = 1:numel(CM.w)
            switch CM.mode(a,b)
                case 1, s = 'step';
                case 2, s = sprintf('%.1f m/s', CM.v(a,b));
                otherwise, s = 'X';
            end
            text(b, a, s, 'HorizontalAlignment', 'center', 'FontSize', 8);
        end
    end
    axis tight;
    xlabel('obstacle width (mm)'); ylabel('obstacle height (mm)');
    title('Capability map: green = step over, yellow = jump (launch speed), red = impossible');
else
    text(0.5, 0.5, 'capability map disabled', 'HorizontalAlignment', 'center');
    axis off;
end

subplot(2,3,5); hold on; box on; grid on;
dE = diff(F(:,10));  md = F(2:end,8);
eR = sum(dE(md ~= 3 & md ~= 8));
eJ = sum(dE(md == 3));
eS = sum(dE(md == 8));
bar(1:3, [eR, eJ, eS]);
for q = 1:3
    v = [eR, eJ, eS];
    text(q, v(q) + 0.02*max(v + 1e-9), sprintf('%.1f J', v(q)), 'HorizontalAlignment', 'center');
end
set(gca, 'XTick', 1:3, 'XTickLabel', {'Walking','Jump crouch','Self-right'});
ylabel('energy (J)');  title('Where the energy goes');

subplot(2,3,6); hold on; box on; grid on;
d = max(F(:,2) - 0.35, 0.5);
plot(F(:,2), F(:,10)./(p.m*p.g*d), 'LineWidth', 1.5);
xlabel('x (m)'); ylabel('cumulative cost of transport (-)');
title('Cost of transport  E / (m g d)');

if exist('sgtitle', 'file')
    sgtitle('Digital Twin V7 - advanced analytics');
end
figure(f3);
end

%% =====================================================================
%%  EXTENSION ANALYTICS : AI, routes, faults, moving-obstacle tracking
%% =====================================================================
function f4 = showExtensions(p, env, S, net)
F = S.F;  t = F(:,1);
f4 = figure('Name','Digital Twin - Extensions (AI, routes, faults, traffic)','NumberTitle','off', ...
            'Position',[130 90 1320 780],'Color','w');

subplot(2,3,1); hold on; box on; grid on;
yyaxis left;   plot(net.loss, 'LineWidth', 1.4);  ylabel('cross-entropy loss');
yyaxis right;  plot(100*net.acc, 'LineWidth', 1.4);  ylabel('validation accuracy (%)');  ylim([0 100]);
xlabel('epoch');  title(sprintf('AI decision network 5-16-3: val. accuracy %.1f %%', 100*net.valAcc));

subplot(2,3,2); hold on; box on;
hg = linspace(0.005, 0.22, 60);  wg = linspace(0.10, 1.0, 50);
Cm = zeros(numel(hg), numel(wg));
for a = 1:numel(hg)
    for b = 1:numel(wg)
        Cm(a,b) = aiDecide(net, hg(a), 0.2, wg(b), 0.95, 1);
    end
end
imagesc(wg, 1000*hg, Cm);
set(gca, 'YDir', 'normal');
colormap(gca, [0.45 0.80 0.45; 0.98 0.82 0.25; 0.45 0.65 0.95]);  caxis([1 3]);
for i = 1:numel(S.ev)
    plot(env.obsHalfW(i), 1000*env.obs(i,3), 'ko', 'MarkerFaceColor', 'w', 'MarkerSize', 8);
    text(env.obsHalfW(i) + 0.02, 1000*env.obs(i,3), sprintf('#%d %s', i, S.ev(i).decision), 'FontSize', 8, 'FontWeight', 'bold');
end
xlabel('obstacle lateral half-width (m)');  ylabel('obstacle height (mm)');
title('AI map: green = STEP, yellow = JUMP, blue = AVOID (w = 200 mm)');

subplot(2,3,3); hold on; box on; grid on;
for i = 1:size(env.obs,1)
    rectangle('Position', [env.obs(i,1), -env.obsHalfW(i), env.obs(i,2), 2*env.obsHalfW(i)], ...
              'FaceColor', [0.55 0.50 0.45], 'EdgeColor', 'k');
    text(env.obs(i,1) + env.obs(i,2)/2, 1.1, sprintf('#%d', i), 'HorizontalAlignment', 'center', 'FontSize', 8);
end
hA = plot(S.R0.shortest.x, S.R0.shortest.y, '--', 'Color', [0.9 0.5 0.1], 'LineWidth', 1.6);
hB = plot(S.R0.safest.x, S.R0.safest.y, '--', 'Color', [0.1 0.6 0.2], 'LineWidth', 1.6);
hT = plot(F(:,2), F(:,17), 'b-', 'LineWidth', 1.8);
lh = [hA, hB, hT];  ln = {'shortest (initial)', 'safest (initial)', 'walked path'};
yl = F(F(:,8) == 11, :);
if ~isempty(yl)
    hY = plot(yl(:,2), yl(:,17), 'r.', 'MarkerSize', 10);
    lh(end+1) = hY;  ln{end+1} = 'yielding to traffic';
end
if env.dynOn
    for i = 1:numel(env.dyn)
        plot(env.dyn(i).x*[1 1], [env.dyn(i).y0, -env.dyn(i).y0], 'm:', 'LineWidth', 1.4);
        if env.dyn(i).rock
            plot(env.dyn(i).x, 0, 'mo', 'MarkerFaceColor', [0.6 0.55 0.5], 'MarkerSize', 6);
        end
    end
end
ylim([-1.3 1.3]);  xlim([0 env.xEnd + 0.5]);
xlabel('x (m)');  ylabel('lateral position y (m)');
title('Path planning: top view (magenta = moving-object corridors, o = moving rocks)');
legend(lh, ln, 'Location', 'southoutside', 'FontSize', 7, 'Orientation', 'horizontal');

subplot(2,3,4); hold on; box on; grid on;
A = S.R0.shortest;  B = S.R0.safest;
bar([A.len B.len; A.E B.E; A.risk B.risk]);
set(gca, 'XTick', 1:3, 'XTickLabel', {'length (m)', 'energy (J)', 'risk score (-)'});
legend('shortest / min-energy', 'safest', 'Location', 'northwest');
title(sprintf('Route comparison (followed: %s, %d re-plans)', upper(env.routeMode), numel(S.routes) - 1));

subplot(2,3,5); hold on; box on; grid on;
fnm = {'LiDAR', 'Leg motors', 'Leg spring', 'IMU'};  fbt = [1 2 4 8];
fcl = [0.9 0.3 0.3; 0.9 0.6 0.1; 0.7 0.3 0.8; 0.2 0.6 0.9];
vtop = max(max(F(:,13))*1.15, 0.5);
hl = [];  hn = {};
for q = 1:4
    on = bitand(F(:,19), fbt(q)) > 0;
    if ~any(on), continue; end
    dd = diff([0; on; 0]);  st = find(dd == 1);  en = find(dd == -1) - 1;
    for s = 1:numel(st)
        patch([t(st(s)) t(en(s)) t(en(s)) t(st(s))], [0 0 vtop vtop], fcl(q,:), 'FaceAlpha', 0.25, ...
              'EdgeColor', 'none', 'HandleVisibility', 'off');
    end
    hl(end+1) = plot(NaN, NaN, 's', 'MarkerFaceColor', fcl(q,:), 'MarkerEdgeColor', 'none'); %#ok<AGROW>
    hn{end+1} = [fnm{q} ' fault']; %#ok<AGROW>
end
hs = plot(t, F(:,13), 'k-', 'LineWidth', 1.3);
legend([hs hl], [{'forward speed'} hn], 'Location', 'northeast', 'FontSize', 7);
xlabel('time (s)');  ylabel('forward speed (m/s)');  ylim([0 vtop]);
title('Fault injection: shaded windows and their effect on speed');

subplot(2,3,6); hold on; box on; grid on;
cc = lines(max(numel(S.dyn), 1));
anyLog = false;
for i = 1:numel(S.dyn)
    Ld = S.dyn(i).log;
    if isempty(Ld), continue; end
    anyLog = true;
    nm = sprintf('#%d', i);
    if env.dyn(i).rock, nm = sprintf('#%d rock', i); end
    plot(Ld(:,1), Ld(:,2), '-', 'Color', cc(i,:), 'LineWidth', 1.8, 'DisplayName', [nm ' true y']);
    plot(Ld(:,1), Ld(:,3), '.', 'Color', cc(i,:), 'MarkerSize', 4, 'DisplayName', [nm ' radar estimate']);
end
if anyLog
    xl = xlim;
    plot(xl, (0.20 + p.robotHW + 0.10)*[1 1], 'r:', 'HandleVisibility', 'off');
    plot(xl, -(0.20 + p.robotHW + 0.10)*[1 1], 'r:', 'HandleVisibility', 'off');
    legend('Location', 'best', 'FontSize', 7);
else
    text(0.5, 0.5, 'moving obstacles disabled', 'HorizontalAlignment', 'center');
end
xlabel('time (s)');  ylabel('lateral position y (m)');
title('Moving objects and rocks: true vs tracked position (red dotted = conflict band)');

if exist('sgtitle', 'file')
    sgtitle('Digital Twin V7 - AI decision, path planning, faults and traffic');
end
figure(f4);
end

%% =====================================================================
%%  SENSOR LOG ANALYTICS
%% =====================================================================
function f5 = showSensorLog(p, env, S) %#ok<INUSL>
F = S.F;  D = S.D;  t = F(:,1);  fm = F(:,19);
f5 = figure('Name','Digital Twin - Sensor log','NumberTitle','off','Position',[160 110 1320 780],'Color','w');
red = [1 0.25 0.2];

subplot(2,3,1); hold on; box on; grid on;
plot(t, D(:,1), '.', 'Color', [0.1 0.6 0.8], 'MarkerSize', 4);
mk = bitand(fm, 1) > 0;
plot(t(mk), D(mk,1), '.', 'Color', red, 'MarkerSize', 5);
xlabel('time (s)');  ylabel('range (m)');  title('LiDAR range to nearest echo (red = LiDAR fault, gaps = dropouts)');

subplot(2,3,2); hold on; box on; grid on;
plot(t, D(:,2), '-', 'Color', [0.2 0.6 0.2]);
mk = bitand(fm, 8) > 0;
plot(t(mk), D(mk,2), '.', 'Color', red, 'MarkerSize', 5);
xlabel('time (s)');  ylabel('deg/s');  title('IMU gyro (red = bias fault)');

subplot(2,3,3); hold on; box on; grid on;
plot(t, D(:,3), '-', 'Color', [0.8 0.6 0.1]);
mk = bitand(fm, 8) > 0;
plot(t(mk), D(mk,3), '.', 'Color', red, 'MarkerSize', 5);
xlabel('time (s)');  ylabel('g');  title('IMU accelerometer (0 g in flight, spikes at landing)');

subplot(2,3,4); hold on; box on; grid on;
yyaxis left;   plot(t, D(:,4), 'LineWidth', 1.5);  ylabel('battery SOC (%)');  ylim([0 105]);
yyaxis right;  plot(t, D(:,5), '-', 'LineWidth', 0.8);  ylabel('leg drive power (W)');
xlabel('time (s)');  title('Battery and leg drive power');

subplot(2,3,5); hold on; box on; grid on;
yyaxis left;   plot(t, D(:,6), 'LineWidth', 1.5);  ylabel('effective spring health (%)');  ylim([50 105]);
yyaxis right;  plot(t, D(:,7), '-', 'LineWidth', 0.8);  ylabel('spring force (N)');
xlabel('time (s)');  title('Spring health (incl. fault loss) and force');

subplot(2,3,6); hold on; box on; grid on;
plot(t, F(:,8), '.', 'MarkerSize', 5);
set(gca, 'YTick', 1:11, 'YTickLabel', {'Walking','Planning','Crouching','Launch','Flight','Landing','Tumble', ...
         'Self-right','End','Detour','Yielding'});
ylim([0.5 11.5]);  xlabel('time (s)');  title('Locomotion mode timeline');

if exist('sgtitle', 'file')
    sgtitle('Digital Twin V7 - sensor log');
end
figure(f5);
end

%% =====================================================================
%%  EXTENSION 6 ANALYTICS : adaptive spring control
%% =====================================================================
function f6 = showAdaptive(p, env, S)
f6 = figure('Name', 'Digital Twin - Adaptive spring control', 'NumberTitle', 'off', ...
            'Position', [190 130 1100 520], 'Color', 'w');
j = find([S.ev.executed]);
subplot(1, 2, 1); hold on; box on; grid on;
if ~isempty(j)
    c0 = zeros(numel(j), 1);  c1 = c0;
    for q = 1:numel(j)
        pl = S.ev(j(q)).plan;  c1(q) = 1000*pl.comp;  c0(q) = c1(q);
        if isfield(pl, 'comp0'), c0(q) = 1000*pl.comp0; end
    end
    bar(1:numel(j), [c0 c1]);
    plot([0.4 numel(j) + 0.6], 1000*p.maxTravel*[1 1], 'r--');
    set(gca, 'XTick', 1:numel(j), 'XTickLabel', cellstr(num2str(j(:))));
    legend('nominal (min-energy)', 'adaptive', 'max travel', 'Location', 'northwest');
end
xlabel('jumped obstacle #'); ylabel('leg crouch (mm)'); title('Adaptive crouch chosen per obstacle');
subplot(1, 2, 2); hold on; box on; grid on;
hh = [0.08 0.10 0.12 0.14 0.16 0.18 0.20];  ww = [0.10 0.20 0.30];  cc = lines(3);  lh = [];  ln = {};
for b = 1:numel(ww)
    a0 = NaN(size(hh));  a1 = a0;
    for a = 1:numel(hh)
        e2 = env;  e2.obs = [1.0 ww(b) hh(a)];
        pl = planJump(p, e2, 1, hh(a), ww(b), 0, 1);
        if pl.feasible
            a0(a) = 1000*pl.comp;
            pl2 = adaptSpring(p, e2, 1, pl, hh(a), ww(b), 0, 1, 1);
            a1(a) = 1000*pl2.comp;
        end
    end
    plot(1000*hh, a0, '--o', 'Color', cc(b,:));
    lh(end+1) = plot(1000*hh, a1, '-s', 'Color', cc(b,:), 'LineWidth', 1.6); %#ok<AGROW>
    ln{end+1} = sprintf('w = %.0f mm (dashed = nominal)', 1000*ww(b)); %#ok<AGROW>
end
legend(lh, ln, 'Location', 'northwest', 'FontSize', 8);
xlabel('obstacle height (mm)'); ylabel('leg crouch (mm)'); title('Crouch vs obstacle size');
if exist('sgtitle', 'file'), sgtitle('Digital Twin V7 - adaptive spring control'); end
figure(f6);
end

%% =====================================================================
%%  EXTENSION 7 ANALYTICS : multi-robot inspection
%% =====================================================================
function f7 = showTeam(p, env, S) %#ok<INUSL>
f7 = NaN;
if isempty(S.M), return; end
M = S.M;  nR = M.nR;  cols = [0.12 0.38 0.80; 0.95 0.45 0.15; 0.25 0.75 0.35];
f7 = figure('Name', 'Digital Twin - Multi-robot inspection', 'NumberTitle', 'off', ...
            'Position', [210 150 1200 720], 'Color', 'w');
subplot(2, 2, 1); hold on; box on; grid on;
for i = 1:size(env.obs, 1)
    rectangle('Position', [env.obs(i,1), -env.obsHalfW(i), env.obs(i,2), 2*env.obsHalfW(i)], ...
              'FaceColor', [0.55 0.50 0.45], 'EdgeColor', 'k');
    text(env.obs(i,1) + env.obs(i,2)/2, 1.12, sprintf('#%d R%d', i, M.owner(i)), 'HorizontalAlignment', 'center', ...
         'FontSize', 8, 'Color', cols(M.owner(i),:)*0.8);
end
lh = [];  ln = {};
for k = 1:nR
    Fk = M.S{k}.F;
    lh(end+1) = plot(Fk(:,2), Fk(:,17), '-', 'Color', cols(k,:), 'LineWidth', 1.8); %#ok<AGROW>
    ln{end+1} = sprintf('R%d', k); %#ok<AGROW>
end
ylim([-1.3 1.3]); xlim([0 env.xEnd + 0.5]); xlabel('x (m)'); ylabel('y (m)');
title('Top view: paths and inspector assignment');
legend(lh, ln, 'Location', 'southoutside', 'Orientation', 'horizontal');

subplot(2, 2, 2); hold on; box on; grid on;
for k = 1:nR
    plot(M.S{k}.F(:,1) + M.t0(k), M.S{k}.F(:,2), '-', 'Color', cols(k,:), 'LineWidth', 1.6);
end
xlabel('mission time (s)'); ylabel('x (m)'); title('Staggered start: progress of each robot');

subplot(2, 2, 3); hold on; box on; grid on;
for q = 1:size(M.pair, 1)
    plot(M.tg, M.sep(q,:), 'LineWidth', 1.4, 'DisplayName', sprintf('R%d-R%d', M.pair(q,1), M.pair(q,2)));
end
plot([M.tg(1) M.tg(end)], 0.35*[1 1], 'r--', 'DisplayName', 'safety distance');
legend('Location', 'best'); xlabel('time (s)'); ylabel('separation (m)'); title('Inter-robot separation');

subplot(2, 2, 4); hold on; box on; grid on;
E = cellfun(@(s) s.E, M.S);  T = cellfun(@(s) s.t, M.S);
bar(1:nR, E);
for k = 1:nR
    text(k, E(k) + 0.03*max(E), sprintf('%.1f J, %.0f s\ninspects %s', E(k), T(k), mat2str(find(M.owner == k)')), ...
         'HorizontalAlignment', 'center', 'FontSize', 8);
end
set(gca, 'XTick', 1:nR, 'XTickLabel', arrayfun(@(k) sprintf('R%d', k), 1:nR, 'UniformOutput', false));
ylim([0 1.35*max(E)]); ylabel('energy (J)'); title('Energy and inspection load per robot');
if exist('sgtitle', 'file'), sgtitle('Digital Twin V7 - multi-robot cooperative inspection'); end
figure(f7);
end

%% =====================================================================
%%  EXTENSION 8 : TERRAIN CLASSIFICATION
%% =====================================================================
function [vm, vs, cm, cs] = terrainModel()
vm = [0.02 0.12 0.07 0.04];  vs = [0.008 0.03 0.020 0.012];
cm = [0.02 0.06 0.08 0.15];  cs = [0.012 0.012 0.012 0.020];
end

function c = zoneClass(env, x)
zmap = [1 1 2 1 3 4  1 1 2 1 3 4];
c = zmap(zoneAt(env, x));
end

function tc = trainTerrainClassifier(p) %#ok<INUSD>
rs = RandStream('mt19937ar', 'Seed', 41);
[vm, vs, cm, cs] = terrainModel();
n = 500;  tc.mu = zeros(4, 2);  tc.sd = zeros(4, 2);
for c = 1:4
    f = [vm(c) + vs(c)*randn(rs, n, 1), cm(c) + cs(c)*randn(rs, n, 1)];
    tc.mu(c,:) = mean(f);  tc.sd(c,:) = std(f);
end
end

function T = buildTerrainLog(p, env, S, tc) %#ok<INUSL>
F = S.F;  N = size(F, 1);
rs = RandStream('mt19937ar', 'Seed', 43);
[vm, vs, cm, cs] = terrainModel();
T = zeros(N, 3);  LL = zeros(1, 4);  pred = 1;  conf = 0.25;
for k = 1:N
    c0 = zoneClass(env, F(k,2));
    if any(F(k,8) == [1 10 11]) && F(k,13) > 0.05
        f = [vm(c0) + vs(c0)*randn(rs), cm(c0) + cs(c0)*randn(rs)];
        ll = -sum(0.5*((repmat(f, 4, 1) - tc.mu)./tc.sd).^2 + log(tc.sd), 2)';
        LL = 0.97*LL + ll;
        e = exp(LL - max(LL));  pr = e/sum(e);
        [conf, pred] = max(pr);
    end
    T(k,:) = [c0 pred conf];
end
end

%% =====================================================================
%%  EXTENSION 9 : COLLISION PREDICTION
%% =====================================================================
function C = buildCollisionLog(p, env, S)
F = S.F;  N = size(F, 1);  Hz = 2.0;  sv = 0:0.05:Hz;  vP = p.vCruise;
C = [99*ones(N,1), zeros(N,1), zeros(N,1), 9*ones(N,1)];
for k = 1:N
    mode = F(k,8);
    if any(mode == [3 4 5 6 7 8 9]), continue; end
    t = F(k,1);  x = F(k,2);  y = F(k,17);
    xr = x + vP*sv;
    ttc = Inf;  who = 0;  mit = false;  gmin = Inf;
    if env.dynOn
        for i = 1:numel(env.dyn)
            d = env.dyn(i);  t0 = S.dyn(i).t0;
            if isnan(t0), tt = zeros(size(sv)); else, tt = max(t + sv - t0, 0); end
            yd = min(max(d.y0 + d.vy*tt, min(d.y0, -d.y0)), max(d.y0, -d.y0));
            ex = max(abs(xr - d.x) - (d.len + p.L)/2, 0);
            ey = max(abs(y - yd) - d.hw - p.robotHW, 0);
            gmin = min(gmin, min(hypot(ex, ey)));
            j = find(ex == 0 & ey == 0, 1);
            if ~isempty(j) && sv(j) < ttc
                ttc = sv(j);  who = i;  mit = (mode == 11);
            end
        end
    end
    for i = 1:size(env.obs, 1)
        ahead = env.obs(i,1) - (x + p.L/2);
        tt = max(ahead, 0)/vP;
        if ahead > -env.obs(i,2) && tt <= Hz && abs(y) < env.obsHalfW(i) + p.robotHW && env.obs(i,3) > p.hRollMax
            if tt < ttc
                ttc = tt;  who = 100 + i;
                mit = S.ev(i).tSense <= t && any(strcmp(S.ev(i).decision, {'JUMP','AVOID'}));
            end
        end
    end
    if isinf(ttc), lv = 0; elseif mit, lv = 1; elseif ttc < 0.7, lv = 3; else, lv = 2; end
    if isinf(ttc), ttc = 99; end
    C(k,:) = [ttc lv who gmin];
end
end

function f8 = showTerrainCollision(p, env, S) %#ok<INUSL>
f8 = figure('Name','Digital Twin - Terrain AI + collision prediction','NumberTitle','off', ...
            'Position',[230 170 1250 700],'Color','w');
F = S.F;  t = F(:,1);  tn = {'Asphalt','Gravel','Grass','Sand'};
subplot(2,2,1); hold on; box on; grid on;
if ~isempty(S.T)
    plot(F(:,2), S.T(:,1), 'k-', 'LineWidth', 2);
    plot(F(:,2), S.T(:,2), 'r.', 'MarkerSize', 5);
    set(gca, 'YTick', 1:4, 'YTickLabel', tn);  ylim([0.5 4.5]);
    legend('true', 'classified', 'Location', 'best');
end
xlabel('x (m)'); title('Terrain classification along the course');
subplot(2,2,2); hold on; box on;
if ~isempty(S.T)
    CMx = zeros(4);
    for k = 1:size(S.T,1), CMx(S.T(k,1), S.T(k,2)) = CMx(S.T(k,1), S.T(k,2)) + 1; end
    CMn = CMx./max(sum(CMx, 2), 1);
    imagesc(CMn); colormap(gca, flipud(gray)); caxis([0 1]);
    set(gca, 'XTick', 1:4, 'XTickLabel', tn, 'YTick', 1:4, 'YTickLabel', tn, 'YDir', 'reverse');
    for a = 1:4
        for b = 1:4, text(b, a, sprintf('%.0f%%', 100*CMn(a,b)), 'HorizontalAlignment', 'center', 'Color', [0.8 0.1 0.1]); end
    end
    xlabel('predicted'); ylabel('true');
    title(sprintf('Confusion matrix (accuracy %.1f %%)', 100*mean(S.T(:,1) == S.T(:,2))));
end
subplot(2,2,3); hold on; box on; grid on;
if ~isempty(S.C)
    ttc = S.C(:,1);  ttc(ttc > 5) = NaN;
    plot(t, ttc, 'b.', 'MarkerSize', 5);
    plot([t(1) t(end)], 0.7*[1 1], 'r--');
    xlabel('time (s)'); ylabel('time-to-collision (s)'); title('Predicted TTC (2 s horizon; red = imminent limit)');
end
subplot(2,2,4); hold on; box on; grid on;
if ~isempty(S.C)
    cl = [0.3 0.8 0.3; 1 0.8 0.2; 1 0.5 0.1; 1 0.1 0.1];
    for lv = 0:3
        m = S.C(:,2) == lv;
        if any(m), plot(t(m), lv*ones(sum(m),1), '.', 'Color', cl(lv+1,:), 'MarkerSize', 6); end
    end
    set(gca, 'YTick', 0:3, 'YTickLabel', {'clear','tracked','predicted','imminent'}); ylim([-0.5 3.5]);
    xlabel('time (s)'); title('Collision risk level timeline');
end
if exist('sgtitle', 'file'), sgtitle('Digital Twin V7 - terrain classification and collision prediction'); end
figure(f8);
end

%% =====================================================================
%%  EXTENSION 10 : OBSTACLE HAZARD RANKING
%% =====================================================================
function Z = hazardRanking(p, env, S) %#ok<INUSL>
n = size(env.obs, 1);
Z.score = zeros(n, 1);  Z.level = zeros(n, 1);  Z.pLand = NaN(n, 1);
for i = 1:n
    e  = S.ev(i);
    h  = env.obs(i,3);  w = env.obs(i,2);  hw = env.obsHalfW(i);
    zi = zoneAt(env, env.obs(i,1));
    switch e.decision
        case 'JUMP',  dc = 1.0;
        case 'AVOID', dc = 0.45;
        otherwise,    dc = 0;
    end
    risk = 0;
    q = find(S.MC.idx == i, 1);
    if ~isempty(q), Z.pLand(i) = S.MC.prob(q);  risk = 1 - S.MC.prob(q); end
    soil = 1 - env.zone.soft(zi);
    Z.score(i) = 40*min(h/0.2, 1) + 15*min(w/0.4, 1) + 10*min(hw/0.9, 1) + 20*dc ...
               + 10*min(5*risk, 1) + 5*min(soil/0.12, 1);
    if Z.score(i) < 25
        Z.level(i) = 1;
    elseif Z.score(i) < 50
        Z.level(i) = 2;
    else
        Z.level(i) = 3;
    end
end
[~, Z.order] = sort(Z.score, 'descend');
Z.names = {'LOW', 'MEDIUM', 'HIGH'};
end

%% =====================================================================
%%  EXTENSION 11 : ENERGY BUDGET + RANGE FORECAST
%% =====================================================================
function R = energyForecast(p, env, S)
F = S.F;  t = F(:,1);  x = cummax(F(:,2));  E = F(:,10);
[xu, iu] = unique(x, 'first');
xq = 0.5:0.5:max(xu);
Eq = interp1(xu, E(iu), xq, 'linear', 'extrap');
tq = interp1(xu, t(iu), xq, 'linear', 'extrap');
nq = numel(xq);
rate = zeros(1, nq);
for j = 2:nq
    j0 = max(j - 4, 1);
    rate(j) = (Eq(j) - Eq(j0))/(xq(j) - xq(j0));
end
rate(1) = rate(2);
Erem = p.Ebat - (Eq - p.solarW*tq);
R.xq = xq;  R.rate = rate;  R.Erem = Erem;
R.range  = Erem./max(rate, 1e-3);
R.back   = xq - 0.35;
R.margin = R.range - R.back;
R.minMargin = min(R.margin);
R.needRTB   = R.minMargin < 0;
nz = numel(env.zone.name);
R.zoneE = zeros(1, nz);  R.zoneD = zeros(1, nz);
for q = 1:nz
    a  = min(max(env.zone.x0(q), xu(1)), xu(end));
    b  = min(max(env.zone.x1(q), xu(1)), xu(end));
    ea = interp1(xu, E(iu), a);  eb = interp1(xu, E(iu), b);
    R.zoneE(q) = max(eb - ea, 0);  R.zoneD(q) = max(b - a, 0);
end
R.zoneJ = R.zoneE./max(R.zoneD, 1e-3);
end

function printNew(p, env, S) %#ok<INUSL>
fprintf(' EXTENSION 10 - OBSTACLE HAZARD RANKING: %s\n', onoff(~isempty(S.HZ)));
if ~isempty(S.HZ)
    Z = S.HZ;
    for q = 1:numel(Z.order)
        i = Z.order(q);
        fprintf('  #%-2d %-15s score %5.1f  %-6s  decision %-6s', i, env.obsName{i}, Z.score(i), Z.names{Z.level(i)}, S.ev(i).decision);
        if ~isnan(Z.pLand(i)), fprintf('  landing success %.1f %%', 100*Z.pLand(i)); end
        fprintf('\n');
    end
    fprintf('  HIGH %d | MEDIUM %d | LOW %d\n', sum(Z.level == 3), sum(Z.level == 2), sum(Z.level == 1));
end
fprintf(' EXTENSION 11 - ENERGY BUDGET AND RANGE FORECAST: %s\n', onoff(~isempty(S.EF)));
if ~isempty(S.EF)
    R = S.EF;
    for q = 1:numel(env.zone.name)
        fprintf('  %-10s %5.1f m  %6.1f J  %5.2f J/m\n', env.zone.name{q}, R.zoneD(q), R.zoneE(q), R.zoneJ(q));
    end
    fprintf('  worst-case return-to-base margin %.1f m -> %s\n', R.minMargin, ...
            ternary(R.needRTB, 'RETURN-TO-BASE REQUIRED', 'battery sufficient to return from anywhere'));
end
fprintf('---------------------------------------------------------------------\n');
end

function f9 = showHazardEnergy(p, env, S)
f9 = figure('Name', 'Digital Twin - Hazard ranking + energy forecast', 'NumberTitle', 'off', ...
            'Position', [250 190 1250 700], 'Color', 'w');
cl = [0.3 0.8 0.3; 1 0.8 0.2; 0.9 0.25 0.2];
subplot(2,2,1); hold on; box on; grid on;
if ~isempty(S.HZ)
    Z = S.HZ;  n = numel(Z.score);
    for i = 1:n
        bar(i, Z.score(i), 'FaceColor', cl(Z.level(i),:));
        text(i, Z.score(i) + 2.5, S.ev(i).decision(1), 'HorizontalAlignment', 'center', 'FontWeight', 'bold');
    end
    plot([0 n+1], [25 25], 'k:');  plot([0 n+1], [50 50], 'k:');
    set(gca, 'XTick', 1:n);  xlim([0 n+1]);  ylim([0 105]);
    xlabel('obstacle #');  ylabel('hazard score (0-100)');
    title('Hazard ranking (S step / J jump / A avoid; green LOW, yellow MEDIUM, red HIGH)');
end
subplot(2,2,2); hold on; box on; grid on;
if ~isempty(S.EF)
    R = S.EF;  nz = numel(env.zone.name);
    bar(1:nz, R.zoneJ);
    set(gca, 'XTick', 1:nz, 'XTickLabel', env.zone.name, 'XTickLabelRotation', 45);
    ylabel('energy per metre (J/m)');  title('Energy cost per terrain zone');
end
subplot(2,2,3); hold on; box on; grid on;
if ~isempty(S.EF)
    plot(R.xq, R.range, 'b-', 'LineWidth', 1.6);
    plot(R.xq, R.back, 'r--', 'LineWidth', 1.4);
    legend('projected range at current consumption', 'distance back to base', 'Location', 'best');
    xlabel('x (m)');  ylabel('metres');
    title(sprintf('Range forecast (worst return margin %.1f m)', R.minMargin));
end
subplot(2,2,4); hold on; box on; grid on;
if ~isempty(S.EF)
    yyaxis left;   plot(R.xq, 100*R.Erem/p.Ebat, 'LineWidth', 1.6);  ylabel('battery left (%)');  ylim([0 105]);
    yyaxis right;  plot(R.xq, R.rate, '-', 'LineWidth', 1.2);  ylabel('consumption (J/m, last 2 m)');
    xlabel('x (m)');  title('Battery and running consumption');
end
if exist('sgtitle', 'file'), sgtitle('Digital Twin V7 - hazard ranking and energy forecast'); end
figure(f9);
end

function appendNewReport(fn, p, env, S) %#ok<INUSL>
fid = fopen(fn, 'a');
if fid < 0, return; end
if ~isempty(S.HZ)
    Z = S.HZ;
    fprintf(fid, '\nEXTENSION 10 - hazard ranking (score, level, decision)\n');
    for q = 1:numel(Z.order)
        i = Z.order(q);
        fprintf(fid, '  #%d %-15s %5.1f %-6s %s\n', i, env.obsName{i}, Z.score(i), Z.names{Z.level(i)}, S.ev(i).decision);
    end
end
if ~isempty(S.EF)
    R = S.EF;
    fprintf(fid, '\nEXTENSION 11 - energy per zone\n');
    for q = 1:numel(env.zone.name)
        fprintf(fid, '  %-10s %5.1f m %6.1f J %5.2f J/m\n', env.zone.name{q}, R.zoneD(q), R.zoneE(q), R.zoneJ(q));
    end
    fprintf(fid, '  worst-case return-to-base margin %.1f m\n', R.minMargin);
end
fclose(fid);
end

%% =====================================================================
%%  AUTO MISSION REPORT (CSV + TXT + PNG)
%% =====================================================================
function exportReport(p, env, S, CM, figs)
try
    stamp = datestr(now, 'yyyymmdd_HHMMSS');
    base  = ['SpringJumper_V7_3D_' stamp];

    fid = fopen([base '_telemetry.csv'], 'w');
    fprintf(fid, 't,x,z,phi_rad,gaitDist_over_r,crouch_m,legExt_m,mode,msgId,E_J,P_W,Fspring_N,vx,vz,active,health,y,yaw_rad,faultMask\n');
    fprintf(fid, [repmat('%.6g,', 1, 18) '%.6g\n'], S.F');
    fclose(fid);

    fid = fopen([base '_sensors.csv'], 'w');
    fprintf(fid, 't,lidar_m,gyro_dps,accel_g,soc_pct,motor_W,spring_health_pct,spring_N\n');
    fprintf(fid, [repmat('%.6g,', 1, 7) '%.6g\n'], [S.F(:,1), S.D]');
    fclose(fid);

    fid = fopen([base '_report.txt'], 'w');
    fprintf(fid, 'SPRING-LEGGED JUMPING ROBOT DIGITAL TWIN V7 (3D) - MISSION REPORT\n');
    fprintf(fid, 'status: %s | time %.1f s | distance %.2f m | energy %.1f J\n', S.status, S.t, S.x, S.E);
    fprintf(fid, 'cost of transport %.2f | battery left %.0f %% | spring health %.1f %% (%d jumps)\n\n', ...
            S.E/(p.m*p.g*max(S.x - 0.35, 0.1)), 100*batterySOC(p, S.t, S.E), 100*S.health, S.jumps);
    fprintf(fid, 'legs: 4 telescopic coil-spring legs, sleeve %.0f mm, coil radius %.0f mm x %d turns, stride %.0f mm, swing lift %.0f mm\n', ...
            1000*p.legL, 1000*p.coilR, p.nCoil, 1000*p.stride, 1000*p.stepH);
    fprintf(fid, 'running jump: run-up speed %.2f m/s, crouch on the move %d frames, no stop before the jump\n', p.vJump, p.nCrouchRun);
    fprintf(fid, 'extensions: AI %s | moving obstacles %s (%d, %d moving rocks) | fault injection %s | route %s | adaptive spring %s | multi-robot %s | terrain AI %s | collision prediction %s\n\n', ...
            onoff(env.aiOn), onoff(env.dynOn), numel(env.dyn), sum([env.dyn.rock]), onoff(env.faultsOn), upper(env.routeMode), ...
            onoff(env.asOn), onoff(env.multiOn), onoff(env.terrOn), onoff(env.colOn));
    for i = 1:numel(S.ev)
        e = S.ev(i);
        fprintf(fid, '#%d %-15s h=%.1f mm (sigma %.2f) w=%.0f mm hw=%.0f mm -> %s', i, env.obsName{i}, ...
                1000*e.hM, 1000*e.sigH, 1000*e.wM, 1000*e.hwM, e.decision);
        if e.cAI > 0
            fprintf(fid, ' | AI %s %.0f %%%s', aiName(e.cAI), 100*e.pAI, ternary(e.override, ' (overridden)', ''));
        end
        fprintf(fid, ' | Estep %s Ejump %s Eavoid %s', fmtE(e.Eroll), fmtE(e.Ejump), fmtE(e.Eavoid));
        if ~isempty(e.plan)
            qq = find(S.MC.idx == i, 1);
            fprintf(fid, ' | v=%.2f m/s, angle %.0f deg, run-up %.2f m/s, crouch %.1f mm, landing %.1f deg', ...
                    e.plan.v, e.plan.theta*180/pi, e.plan.vApp, 1000*e.plan.comp, e.landPhi);
            if isfield(e.plan, 'comp0')
                fprintf(fid, ' (adapted from %.1f mm, x%.2f)', 1000*e.plan.comp0, e.plan.asMul);
            end
            if ~isempty(qq), fprintf(fid, ', MC success %.1f %%', 100*S.MC.prob(qq)); end
        end
        fprintf(fid, '\n');
    end
    fprintf(fid, '\nroute (initial): shortest %.2f m / %.1f J / risk %.1f ; safest %.2f m / %.1f J / risk %.1f ; %d re-plans\n', ...
            S.R0.shortest.len, S.R0.shortest.E, S.R0.shortest.risk, S.R0.safest.len, S.R0.safest.E, S.R0.safest.risk, ...
            numel(S.routes) - 1);
    if env.faultsOn
        for i = 1:numel(env.faults)
            fprintf(fid, 'fault %-6s x = %.1f ... %.1f m\n', env.faults(i).type, env.faults(i).x0, env.faults(i).x1);
        end
    end
    if env.dynOn
        fprintf(fid, 'time spent yielding to moving objects and rocks: %.1f s\n', sum(S.F(:,8) == 11)*0.01);
        for i = 1:numel(env.dyn)
            if env.dyn(i).rock
                fprintf(fid, 'moving rock #%d (%s) at x = %.1f m, triggered at t = %.2f s\n', i, env.dyn(i).name, env.dyn(i).x, S.dyn(i).t0);
            end
        end
    end
    if ~isempty(S.T)
        fprintf(fid, 'terrain classification: online accuracy %.1f %%\n', 100*mean(S.T(:,1) == S.T(:,2)));
    end
    if ~isempty(S.C)
        fprintf(fid, 'collision prediction: predicted-collision frames %d, imminent frames %d\n', ... 
                sum(S.C(:,2) >= 2), sum(S.C(:,2) == 3));
    end
    if ~isempty(S.M)
        M = S.M;
        fprintf(fid, '\nmulti-robot team: %d robots, start stagger %.1f s\n', M.nR, M.t0(2));
        for k = 1:M.nR
            fprintf(fid, '  R%d lane %+.2f m: %s, %.1f J, %.1f s, inspects obstacles %s\n', k, M.lane(k), ...
                    M.S{k}.status, M.S{k}.E, M.S{k}.t, mat2str(find(M.owner == k)'));
        end
        for q = 1:size(M.pair, 1)
            fprintf(fid, '  min separation R%d-R%d: %.2f m\n', M.pair(q,1), M.pair(q,2), M.minSep(q));
        end
    end
    if ~isempty(CM)
        fprintf(fid, '\nCapability map: %d jumpable sizes of %d tested above step-over limit.\n', ...
                sum(CM.mode(:) == 2), sum(CM.mode(:) >= 2));
    end

    fclose(fid);
    appendNewReport([base '_report.txt'], p, env, S);

    names = {'analytics', 'advanced', 'extensions', 'sensors', 'adaptive', 'terrain_collision', 'team', 'hazard_energy'};
    for q = 1:numel(figs)
        if ~isempty(figs(q)) && ishandle(figs(q)), saveas(figs(q), [base '_' names{q} '.png']); end
    end
    fprintf('\n Report written: %s_telemetry.csv, %s_sensors.csv, %s_report.txt, %s_*.png\n', base, base, base, base);
catch ME
    warning('Report export failed: %s', ME.message);
end
end