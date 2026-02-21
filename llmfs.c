#include <lib9.h>
#include <styxserver.h>
#define TESTING
#include "run.c"

/*
 * An in-memory file server
 * allowing truncation, removal on closure, wstat and
 * all other file operations
 */

char *fsremove(Qid);

enum {
	StateIdle,
	StatePrompting,
	StateGenerating,
	StateEOF
};

typedef struct LlmConn LlmConn;
struct LlmConn {
	int id;
	int opens;
	int state;
	
	char *prompt_buf;
	int prompt_len;
	int prompt_size;

	char *output_buf;
	int output_len;
	int read_pos;

	Transformer transformer;
	Sampler sampler;
	
	int *prompt_tokens;
	int num_prompt_tokens;
	int pos;
	int token;
	int next;
	
	LlmConn *next_conn;
};

Styxserver *server;
LlmConn *conns;
int next_conn_id = 1;
Tokenizer tokenizer;

Transformer global_transformer; // initialized once to get config, but each connection might need its own state buffer. Actually, run_state has buffers, we should build_transformer for each connection but point to the same memory mapped weights? 
// run.c's build_transformer does both. We can just use memory mapped weights.
// Let's just build_transformer per connection for now. It memory maps the same file, which the OS will share.
char *checkpoint_path;
char *tokenizer_path = "tokenizer.bin";
float global_temp = 1.0f;
float global_topp = 0.9f;

LlmConn*
getconn(int id)
{
	LlmConn *c;
	for(c = conns; c != nil; c = c->next_conn)
		if(c->id == id)
			return c;
	return nil;
}

void
freeconn(LlmConn *c)
{
	LlmConn **l;
	for(l = &conns; *l != nil; l = &(*l)->next_conn){
		if(*l == c){
			*l = c->next_conn;
			break;
		}
	}
	free_sampler(&c->sampler);
	free_transformer(&c->transformer);
	free(c->prompt_buf);
	free(c->output_buf);
	free(c->prompt_tokens);
	free(c);
}

LlmConn*
newconn(void)
{
	LlmConn *c;
	
	c = calloc(1, sizeof(LlmConn));
	c->id = next_conn_id++;
	c->state = StateIdle;
	c->next_conn = conns;
	conns = c;
	
	build_transformer(&c->transformer, checkpoint_path);
	build_sampler(&c->sampler, c->transformer.config.vocab_size, global_temp, global_topp, 0); // time(nil) for seed? we can just use 0
	
	return c;
}

#define Qclone 1
#define Qllm 2
// we will use connections as ID directly.

char*
fsopen(Qid *qid, int mode)
{
	Styxfile *f;
	LlmConn *c;
	char buf[32];
	int id;

	if(qid->path == Qclone){
		c = newconn();
		id = c->id;
		snprint(buf, sizeof(buf), "%d", id);
		
		int baseid = id * 10;
		#define Qroot 0
		styxadddir(server, Qroot, baseid, buf, 0777|DMDIR, "inferno");
		styxaddfile(server, baseid, baseid+1, "ctl", 0666, "inferno");
		styxaddfile(server, baseid, baseid+2, "data", 0666, "inferno");
		styxaddfile(server, baseid, baseid+3, "status", 0444, "inferno");
		
		f = styxfindfile(server, baseid+1);
		// We just added cf directly and got baseid+1
		if(f != nil)
			*qid = f->d.qid;
		
		c->opens++;
		return nil;
	}

	f = styxfindfile(server, qid->path);
	if(f == nil) return "file not found";
	
	if(strcmp(f->d.name, "ctl") == 0 || strcmp(f->d.name, "data") == 0) {
		id = f->parent->d.qid.path / 10;
		c = getconn(id);
		if(c) c->opens++;
	}

	if(mode&OTRUNC){	/* truncate on open */
		styxfree(f->u);
		f->u = nil;
		f->d.length = 0;
	}
	return nil;
}

char*
fsclose(Qid qid, int mode)
{
	Styxfile *f;
	LlmConn *c;
	int id;

	f = styxfindfile(server, qid.path);
	if(f && (strcmp(f->d.name, "ctl") == 0 || strcmp(f->d.name, "data") == 0)) {
		id = f->parent->d.qid.path / 10;
		c = getconn(id);
		if(c) {
			c->opens--;
			if(strcmp(f->d.name, "data") == 0 && (mode & OWRITE)) {
				if(c->state == StatePrompting) {
					c->state = StateGenerating;
					// reset generation state
					c->pos = 0;
					c->num_prompt_tokens = 0;
					free(c->prompt_tokens);
					c->prompt_tokens = malloc((c->prompt_len + 3) * sizeof(int));
					encode(&tokenizer, c->prompt_buf, 1, 0, c->prompt_tokens, &c->num_prompt_tokens);
					c->token = c->prompt_tokens[0];
				}
			}
			if(c->opens == 0) {
				// do nothing, let fsremove handle it so we can use disjoint commands in sh
			}
		}
	}

	if(mode&ORCLOSE)	/* remove on close */
		return fsremove(qid);
	return nil;
}

char *
fscreate(Qid *qid, char *name, int perm, int mode)
{
	int isdir;
	Styxfile *f;

	USED(mode);
	isdir = perm&DMDIR;
	if(isdir)
		f = styxadddir(server, qid->path, -1, name, perm, "inferno");
	else
		f = styxaddfile(server, qid->path, -1, name, perm, "inferno");
	if(f == nil)
		return Eexist;
	f->u = nil;
	f->d.length = 0;
	*qid = f->d.qid;
	return nil;
}

char *
fsremove(Qid qid)
{
	Styxfile *f;

	f = styxfindfile(server, qid.path);
	if((f->d.qid.type&QTDIR) && f->child != nil) {
		// allow removing non-empty connection directory by cleaning children
		int id = f->d.qid.path / 10;
		LlmConn *c = getconn(id);
		if(c) {
            freeconn(c);
		    // styxrmfile recursively removes, so child check bypass is fine
        } else {
		    return "directory not empty";
        }
    }
	styxfree(f->u);
	styxrmfile(server, qid.path);	
	return nil;
}

char *
fsread(Qid qid, char *buf, ulong *n, vlong off)
{
	int m;
	Styxfile *f;
	LlmConn *c;
	int id;
	char *str;

	f = styxfindfile(server, qid.path);
	if(f == nil) return "file not found";

	if(strcmp(f->d.name, "ctl") == 0) {
		id = f->parent->d.qid.path / 10;
		char tmpbuf[32];
		snprint(tmpbuf, sizeof(tmpbuf), "%d", id);
		m = strlen(tmpbuf);
		if(off >= m) *n = 0;
		else {
			if(off + *n > m) *n = m - off;
			memmove(buf, tmpbuf + off, *n);
		}
		return nil;
	} else if(strcmp(f->d.name, "status") == 0) {
		id = f->parent->d.qid.path / 10;
		c = getconn(id);
		if(!c) return "connection closed";
		char tmpbuf[128];
		char *state = "Idle";
		if(c->state == StatePrompting) state = "Prompting";
		else if(c->state == StateGenerating) state = "Generating";
		else if(c->state == StateEOF) state = "EOF";
		snprint(tmpbuf, sizeof(tmpbuf), "cmd/%d %d %s /n/llm/%d llm", c->id, c->opens, state, c->id);
		m = strlen(tmpbuf);
		if(off >= m) *n = 0;
		else {
			if(off + *n > m) *n = m - off;
			memmove(buf, tmpbuf + off, *n);
		}
		return nil;
	} else if(strcmp(f->d.name, "data") == 0) {
		id = f->parent->d.qid.path / 10;
		c = getconn(id);
		if(!c) return "connection closed";

		if(c->state == StateIdle) return "no prompt";
		
		int bytes_read = 0;
		char *p = buf;
		while(bytes_read < *n) {
			if(c->read_pos < c->output_len) {
				int amt = c->output_len - c->read_pos;
				if(amt > (*n - bytes_read)) amt = *n - bytes_read;
				memmove(p, c->output_buf + c->read_pos, amt);
				p += amt;
				bytes_read += amt;
				c->read_pos += amt;
			} else {
				if(c->state == StateGenerating) {
					// generate next token
					float *logits = forward(&c->transformer, c->token, c->pos);
					if(c->pos < c->num_prompt_tokens - 1) {
						c->next = c->prompt_tokens[c->pos + 1];
					} else {
						c->next = sample(&c->sampler, logits);
					}
					c->pos++;
					
					if(c->next == 1 || c->next == 2) { // BOS shouldn't happen usually but acts as EOF, 2 is EOS
						c->state = StateEOF;
						break;
					}
					
					char *piece = decode(&tokenizer, c->token, c->next);
					c->token = c->next;
					
					if(piece) {
						int plen = strlen(piece);
						if(plen > 0) {
							c->output_buf = realloc(c->output_buf, c->output_len + plen + 1);
							memmove(c->output_buf + c->output_len, piece, plen + 1);
							c->output_len += plen;
						}
					}
				} else if(c->state == StateEOF) {
					break;
				}
			}
		}
		*n = bytes_read;
		return nil;
	}

	// default file handling
	m = f->d.length;
	if(off >= m)
		*n = 0;
	else{
		if(off + *n > m)
			*n = m-off;
		memmove(buf, (char*)f->u+off, *n);
	}
	return nil;
}

char*
fswrite(Qid qid, char *buf, ulong *n, vlong off)
{
	Styxfile *f;
	vlong m, p;
	char *u;
	int id;
	LlmConn *c;

	f = styxfindfile(server, qid.path);
	if(f == nil) return "file not found";

	if(strcmp(f->d.name, "ctl") == 0) {
		id = f->parent->d.qid.path / 10;
		c = getconn(id);
		if(!c) return "connection closed";
		
		char *cmd = malloc(*n + 1);
		memmove(cmd, buf, *n);
		cmd[*n] = '\0';
		
		if(strncmp(cmd, "temp ", 5) == 0) {
			c->sampler.temperature = atof(cmd + 5);
		} else if(strncmp(cmd, "top ", 4) == 0) {
			c->sampler.topp = atof(cmd + 4);
		} else {
			free(cmd);
			return "unknown ctl command";
		}
		free(cmd);
		return nil;
	} else if(strcmp(f->d.name, "data") == 0) {
		id = f->parent->d.qid.path / 10;
		c = getconn(id);
		if(!c) return "connection closed";
		
		if(c->state != StatePrompting) {
			c->state = StatePrompting;
			c->prompt_len = 0;
		}
		
		if(c->prompt_len + *n + 1 > c->prompt_size) {
			c->prompt_size = c->prompt_len + *n + 1024;
			c->prompt_buf = realloc(c->prompt_buf, c->prompt_size);
		}
		memmove(c->prompt_buf + c->prompt_len, buf, *n);
		c->prompt_len += *n;
		c->prompt_buf[c->prompt_len] = '\0';
		
		return nil;
	}

	m = f->d.length;
	p = off + *n;
	if(p > m){	/* just grab a larger piece of memory */
		u = styxmalloc(p);
		if(u == nil)
			return "out of memory";
		memset(u, 0, p);
		memmove(u, f->u, m);
		styxfree(f->u);
		f->u = u;
		f->d.length = p;
	}
	memmove((char*)f->u+off, buf, *n);
	return nil;
}

char*
fswstat(Qid qid, Dir *d)
{
	Styxfile *f, *tf;
	Client *c;
	int owner;

	/* the most complicated operation when fully allowed */

	c = styxclient(server);
	f = styxfindfile(server, qid.path);
	owner = strcmp(c->uname, f->d.uid) == 0;
	if(d->name != nil && strcmp(d->name, f->d.name) != 0){
		/* need write permission in parent directory */
		if(!styxperm(f->parent, c->uname, OWRITE))
			return Eperm;
		if((tf = styxaddfile(server, f->parent->d.qid.path, -1, d->name, 0, "")) == nil){
			/* file with same name exists */
			return Eexist;
		}
		else{
			/* undo above addfile */
			styxrmfile(server, tf->d.qid.path);
		}
		/* ok to change name now */
		styxfree(f->d.name);
		f->d.name = strdup(d->name);	
	}
	if(d->uid != nil && strcmp(d->uid, f->d.uid) != 0){
		if(!owner)
			return Eperm;
		styxfree(f->d.uid);
		f->d.uid = strdup(d->uid);
	}
	if(d->gid != nil && strcmp(d->gid, f->d.gid) != 0){
		if(!owner)
			return Eperm;
		styxfree(f->d.gid);
		f->d.gid = strdup(d->gid);
	}
	if(d->mode != ~0 && d->mode != f->d.mode){
		if(!owner)
			return Eperm;
		if((d->mode&DMDIR) != (f->d.mode&DMDIR))
			return Eperm;	/* cannot change file->directory or vice-verse */
		f->d.mode = d->mode;
	}
	if(d->mtime != ~0 && d->mtime != f->d.mtime){
		if(!owner)
			return Eperm;
		f->d.mtime = d->mtime;
	}
	/* all other file attributes cannot be changed by wstat */
	return nil;
}

Styxops ops = {
	nil,			/* newclient */
	nil,			/* freeclient */

	nil,			/* attach */
	nil,			/* walk */
	fsopen,		/* open */
	fscreate,		/* create */
	fsread,		/* read */
	fswrite,		/* write */
	fsclose,		/* close */
	fsremove,	/* remove */
	nil,			/* stat */
	fswstat,		/* wstat */
};

void
main(int argc, char **argv)
{
	Styxserver s;
	char *port = "6701";

	printf("Starting main\n");
	if(argc >= 2) {
		checkpoint_path = argv[1];
	} else {
		fprintf(stderr, "Usage: llmfs <checkpoint> [options]\n");
		exits("usage");
	}
	
	printf("Building temp_transformer with %s\n", checkpoint_path);
	Transformer temp_transformer;
	build_transformer(&temp_transformer, checkpoint_path);
	printf("Building tokenizer\n");
	build_tokenizer(&tokenizer, tokenizer_path, temp_transformer.config.vocab_size);
	printf("Freeing temp_transformer\n");
	free_transformer(&temp_transformer);

	printf("Initializing styx\n");
	server = &s;
	styxdebug();
	styxinit(&s, &ops, port, 0777, 1);
	
	int rootpath = 0; // styx server sets root as Path 0. But let's verify. According to libstyx Qroot is 0.
	#define Qroot 0
	styxaddfile(&s, Qroot, Qclone, "clone", 0666, "inferno");

	for(;;){
		styxwait(&s);
		styxprocess(&s);
	}
	exits(nil);
}

#undef malloc
#undef free
#undef calloc
#undef realloc

void* kmalloc(size_t size) { return malloc(size); }
void kfree(void *ptr) { free(ptr); }
void* kcalloc(size_t nmemb, size_t size) { return calloc(nmemb, size); }
void* krealloc(void *ptr, size_t size) { return realloc(ptr, size); }

